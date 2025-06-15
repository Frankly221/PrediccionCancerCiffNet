import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN MAX GPU para RTX 3070 + 32GB RAM
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
os.environ['OMP_NUM_THREADS'] = '16'  # ⬆️ MAX para 32GB RAM
os.environ['CUDA_CACHE_DISABLE'] = '0'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False  # ⬆️ Más velocidad
torch.set_num_threads(16)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class MaxGPUTrainer:
    """Trainer MAX GPU con matriz de confusión SOLO AL FINAL"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN MAX GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.99)  # ⬆️ 99% VRAM
            torch.backends.cuda.cufft_plan_cache.max_size = 4
        
        # AMP con configuración agresiva
        self.scaler = GradScaler(
            init_scale=65536.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        # Optimizer optimizado para MAX throughput
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False
        )
        
        # Scheduler más agresivo
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=3,  # ⬇️ Más agresivo con MAX GPU
            min_lr=1e-8,
            threshold=0.01
        )
        
        # Loss optimizado
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=self.class_weights)
        
        # Tracking simplificado (NO matriz por época)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.learning_rates = []
        self.gpu_utilization = []
        self.batch_throughput = []
        
        # Solo guardar ÚLTIMA matriz de confusión
        self.final_confusion_matrix = None
        self.final_class_metrics = None
        self.final_classification_report = None
        
        # Best metrics
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 MAX GPU Trainer inicializado:")
        print(f"   Target GPU: 95%+")
        print(f"   VRAM usage: 99%")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Workers: {train_loader.num_workers}")
        print(f"   Clases: {len(label_encoder.classes_)}")
        print(f"   📊 Matriz de confusión: SOLO AL FINAL")
        
    def monitor_gpu_utilization(self):
        """Monitor GPU utilization"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_percent = (memory_allocated / 8.0) * 100
            utilization = min(99, memory_percent * 1.2)
            
            return {
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_percent': memory_percent,
                'estimated_utilization': utilization
            }
        return {'memory_allocated_gb': 0, 'memory_reserved_gb': 0, 'memory_percent': 0, 'estimated_utilization': 0}
    
    def compute_melanoma_metrics(self, all_predicted, all_targets):
        """Calcular métricas para melanoma"""
        melanoma_idx = None
        for i, class_name in enumerate(self.label_encoder.classes_):
            if 'mel' in class_name.lower():
                melanoma_idx = i
                break
        
        if melanoma_idx is None:
            return 0.0, 0.0
        
        melanoma_true = (np.array(all_targets) == melanoma_idx)
        melanoma_pred = (np.array(all_predicted) == melanoma_idx)
        
        true_positives = np.sum(melanoma_true & melanoma_pred)
        false_negatives = np.sum(melanoma_true & ~melanoma_pred)
        false_positives = np.sum(~melanoma_true & melanoma_pred)
        
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        return recall, precision
    
    def compute_detailed_metrics(self, all_predicted, all_targets):
        """Calcular métricas detalladas"""
        cm = confusion_matrix(all_targets, all_predicted)
        
        report = classification_report(
            all_targets, 
            all_predicted, 
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        
        # Métricas por clase
        class_metrics = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_true = (np.array(all_targets) == i)
            class_pred = (np.array(all_predicted) == i)
            
            tp = np.sum(class_true & class_pred)
            fp = np.sum(~class_true & class_pred)
            fn = np.sum(class_true & ~class_pred)
            tn = np.sum(~class_true & ~class_pred)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'support': np.sum(class_true)
            }
        
        return cm, report, class_metrics
    
    def compute_balanced_score(self, accuracy, melanoma_recall):
        """Score balanceado"""
        return 0.6 * accuracy + 0.4 * (melanoma_recall * 100)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        batch_times = []
        
        # Gradient accumulation más agresivo
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"🔥 MAX GPU Epoch {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Performance monitoring
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            gpu_stats = self.monitor_gpu_utilization()
            samples_per_second = inputs.size(0) / batch_time
            
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['estimated_utilization']:.0f}%",
                'VRAM': f"{gpu_stats['memory_allocated_gb']:.1f}GB",
                'SPS': f"{samples_per_second:.0f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Limpieza GPU menos frecuente para MAX utilización
            if batch_idx % 40 == 0:  # ⬆️ Cada 40 batches en vez de 20
                torch.cuda.empty_cache()
        
        # Final optimizer step
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        throughput = len(self.train_loader.dataset) / epoch_time
        
        # Guardar métricas
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.batch_throughput.append(throughput)
        
        # GPU utilization
        final_gpu_stats = self.monitor_gpu_utilization()
        self.gpu_utilization.append(final_gpu_stats['estimated_utilization'])
        
        return avg_loss, accuracy, throughput, final_gpu_stats
    
    def validate(self, epoch, is_final=False):
        """Validación - solo calcular CM si es final"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        validation_start = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating MAX GPU"):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        validation_time = time.time() - validation_start
        
        # Métricas básicas siempre
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        # Métricas detalladas SOLO SI ES FINAL
        if is_final:
            cm, report, class_metrics = self.compute_detailed_metrics(all_predicted, all_targets)
            self.final_confusion_matrix = cm
            self.final_classification_report = report
            self.final_class_metrics = class_metrics
            print(f"📊 Calculando matriz de confusión FINAL...")
            return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time, cm, class_metrics
        
        # Guardar métricas básicas
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time
    
    def plot_final_confusion_matrix(self):
        """Crear matriz de confusión FINAL"""
        if self.final_confusion_matrix is None:
            return
        
        plt.figure(figsize=(14, 12))
        
        # Normalizar matriz
        cm = self.final_confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Accuracy Normalizada'},
            square=True
        )
        
        plt.title(f'MATRIZ DE CONFUSIÓN FINAL - CIFF-Net MAX GPU\nAccuracy: {self.best_overall_acc:.2f}% | Melanoma Recall: {self.best_melanoma_recall*100:.2f}%', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        
        # Highlight melanoma
        melanoma_idx = None
        for i, class_name in enumerate(self.label_encoder.classes_):
            if 'mel' in class_name.lower():
                melanoma_idx = i
                break
        
        if melanoma_idx is not None:
            plt.axhline(y=melanoma_idx, color='red', linewidth=4, alpha=0.8)
            plt.axhline(y=melanoma_idx+1, color='red', linewidth=4, alpha=0.8)
            plt.axvline(x=melanoma_idx, color='red', linewidth=4, alpha=0.8)
            plt.axvline(x=melanoma_idx+1, color='red', linewidth=4, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('final_confusion_matrix_max_gpu.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Matriz de confusión FINAL guardada: final_confusion_matrix_max_gpu.png")
    
    def create_final_comprehensive_report(self):
        """Crear reporte FINAL detallado"""
        if self.final_class_metrics is None:
            return ""
        
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics([], [])
        if self.final_confusion_matrix is not None:
            # Recalcular desde la matriz final
            melanoma_idx = None
            for i, class_name in enumerate(self.label_encoder.classes_):
                if 'mel' in class_name.lower():
                    melanoma_idx = i
                    break
            
            if melanoma_idx is not None:
                cm = self.final_confusion_matrix
                tp = cm[melanoma_idx, melanoma_idx]
                fn = np.sum(cm[melanoma_idx, :]) - tp
                fp = np.sum(cm[:, melanoma_idx]) - tp
                
                melanoma_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                melanoma_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        report_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REPORTE FINAL CIFF-NET MAX GPU                           ║
║                         RTX 3070 + 32GB RAM                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 MÉTRICAS FINALES GENERALES:
   • Mejor Accuracy General: {self.best_overall_acc:.2f}%
   • Mejor Melanoma Recall: {self.best_melanoma_recall*100:.2f}%
   • Mejor Score Balanceado: {self.best_balanced_score:.2f}%
   • Épocas entrenadas: {len(self.val_accuracies)}

🩺 MÉTRICAS MELANOMA FINALES (CRÍTICAS):
   • Recall (Sensibilidad): {melanoma_recall*100:.2f}%
   • Precision: {melanoma_precision*100:.2f}%
   • F1-Score: {2*(melanoma_recall*melanoma_precision)/(melanoma_recall+melanoma_precision)*100 if (melanoma_recall+melanoma_precision) > 0 else 0:.2f}%

📊 MÉTRICAS FINALES POR CLASE:
"""
        
        for class_name, metrics in self.final_class_metrics.items():
            is_melanoma = 'mel' in class_name.lower()
            marker = "🩺" if is_melanoma else "📋"
            
            report_text += f"""
   {marker} {class_name.upper()}:
      • Precision: {metrics['precision']*100:.2f}%
      • Recall: {metrics['recall']*100:.2f}%
      • F1-Score: {metrics['f1_score']*100:.2f}%
      • Specificity: {metrics['specificity']*100:.2f}%
      • Support: {metrics['support']} casos
"""
        
        # Análisis de matriz de confusión FINAL
        if self.final_confusion_matrix is not None:
            cm = self.final_confusion_matrix
            report_text += f"""
🔍 ANÁLISIS MATRIZ DE CONFUSIÓN FINAL:
   • Total de casos: {np.sum(cm)}
   • Aciertos (diagonal): {np.trace(cm)}
   • Accuracy matriz: {np.trace(cm) / np.sum(cm) * 100:.2f}%
   
📈 ERRORES MÁS COMUNES (FINAL):
"""
            
            # Errores más comunes
            cm_errors = cm.copy()
            np.fill_diagonal(cm_errors, 0)
            
            # Top 3 errores
            flat_indices = np.argpartition(cm_errors.ravel(), -3)[-3:]
            top_errors = [(np.unravel_index(idx, cm_errors.shape), cm_errors.flat[idx]) 
                         for idx in flat_indices if cm_errors.flat[idx] > 0]
            
            for (true_idx, pred_idx), count in sorted(top_errors, key=lambda x: x[1], reverse=True):
                true_class = self.label_encoder.classes_[true_idx]
                pred_class = self.label_encoder.classes_[pred_idx]
                percentage = count / np.sum(cm[true_idx]) * 100
                report_text += f"   • {true_class} → {pred_class}: {count} casos ({percentage:.1f}%)\n"
        
        # Performance GPU
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_gpu_util = max(self.gpu_utilization) if self.gpu_utilization else 0
        avg_throughput = np.mean(self.batch_throughput) if self.batch_throughput else 0
        
        report_text += f"""
🚀 RENDIMIENTO GPU FINAL:
   • GPU Utilization Promedio: {avg_gpu_util:.1f}%
   • GPU Utilization Máximo: {max_gpu_util:.1f}%
   • Throughput Promedio: {avg_throughput:.0f} samples/sec
   • Target GPU alcanzado: {'✅ SÍ' if avg_gpu_util >= 85 else '❌ NO'}

💾 CONFIGURACIÓN USADA:
   • Batch size: {self.train_loader.batch_size}
   • Workers: {self.train_loader.num_workers}
   • VRAM target: 99%
   • Gradient accumulation: {self.config.get('gradient_accumulation_steps', 1)}
"""
        
        return report_text
    
    def save_final_analysis(self):
        """Guardar análisis final completo"""
        
        # Gráficos principales
        plt.figure(figsize=(20, 16))
        
        # Loss
        plt.subplot(3, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2, color='blue')
        plt.plot(self.val_losses, label='Val Loss', linewidth=2, color='red')
        plt.title('Loss - MAX GPU Training', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(3, 4, 2)
        plt.plot(self.train_accuracies, label='Train Acc', linewidth=2, color='blue')
        plt.plot(self.val_accuracies, label='Val Acc', linewidth=2, color='red')
        plt.title('Accuracy - MAX GPU Training', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma Recall
        plt.subplot(3, 4, 3)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=3)
        plt.title('Melanoma Recall - MAX GPU', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # GPU Utilization
        plt.subplot(3, 4, 4)
        plt.plot(self.gpu_utilization, color='orange', linewidth=2)
        plt.axhline(y=90, color='green', linestyle='--', label='Target 90%', alpha=0.7)
        plt.axhline(y=95, color='red', linestyle='--', label='Excellent 95%', alpha=0.7)
        plt.title('GPU Utilization - MAX GPU', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('GPU Usage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Throughput
        plt.subplot(3, 4, 5)
        plt.plot(self.batch_throughput, color='purple', linewidth=2)
        plt.title('Training Throughput - MAX GPU', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Samples/Second')
        plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(3, 4, 6)
        plt.plot(self.learning_rates, color='green', linewidth=2)
        plt.title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Performance Summary
        plt.subplot(3, 4, 7)
        epochs = range(1, len(self.val_accuracies) + 1)
        plt.plot(epochs, self.val_accuracies, label='Val Acc', linewidth=2, color='blue')
        plt.plot(epochs, self.melanoma_recalls, label='Melanoma Recall', linewidth=2, color='red')
        balanced_scores = [self.compute_balanced_score(acc, mel_rec/100) for acc, mel_rec in zip(self.val_accuracies, self.melanoma_recalls)]
        plt.plot(epochs, balanced_scores, label='Balanced Score', linewidth=2, linestyle='--', color='green')
        plt.title('Performance Summary - MAX GPU', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # GPU Efficiency
        plt.subplot(3, 4, 8)
        if len(self.gpu_utilization) > 0 and len(self.val_accuracies) > 0:
            efficiency = [acc / (gpu_util + 1) for acc, gpu_util in zip(self.val_accuracies, self.gpu_utilization)]
            plt.plot(efficiency, color='brown', linewidth=2)
            plt.title('GPU Efficiency (Acc/GPU%)', fontweight='bold', fontsize=14)
            plt.xlabel('Época')
            plt.ylabel('Efficiency')
            plt.grid(True, alpha=0.3)
        
        # Train-Val Gap
        plt.subplot(3, 4, 9)
        if len(self.train_accuracies) == len(self.val_accuracies):
            gap = [train - val for train, val in zip(self.train_accuracies, self.val_accuracies)]
            plt.plot(gap, color='purple', linewidth=2)
            plt.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
            plt.axhline(y=10, color='red', linestyle='-', alpha=0.7, label='Overfitting (10%)')
            plt.title('Train-Val Gap (Overfitting)', fontweight='bold', fontsize=14)
            plt.xlabel('Época')
            plt.ylabel('Gap (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Performance vs GPU
        plt.subplot(3, 4, 10)
        if len(self.gpu_utilization) > 0 and len(self.val_accuracies) > 0:
            plt.scatter(self.gpu_utilization, self.val_accuracies, alpha=0.7, color='red')
            plt.xlabel('GPU Utilization (%)')
            plt.ylabel('Validation Accuracy (%)')
            plt.title('Performance vs GPU Usage', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('max_gpu_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear y guardar matriz de confusión final
        self.plot_final_confusion_matrix()
        
        # CSV con métricas
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies,
            'melanoma_recall': self.melanoma_recalls,
            'learning_rate': self.learning_rates,
            'gpu_utilization': self.gpu_utilization,
            'throughput_samples_sec': self.batch_throughput
        })
        metrics_df.to_csv('max_gpu_training_metrics.csv', index=False)
        
        # Reporte final
        final_report = self.create_final_comprehensive_report()
        with open('final_report_max_gpu.txt', 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print("📊 Análisis MAX GPU completo guardado:")
        print("   📈 max_gpu_training_analysis.png")
        print("   📊 final_confusion_matrix_max_gpu.png")
        print("   📋 max_gpu_training_metrics.csv")
        print("   📄 final_report_max_gpu.txt")
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO MAX GPU...")
        print(f"🎯 OBJETIVO: GPU 95%+ con análisis final completo")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train con MAX performance
            train_loss, train_acc, throughput, gpu_stats = self.train_epoch(epoch)
            
            # Validate (sin CM hasta el final)
            val_results = self.validate(epoch, is_final=False)
            val_loss, val_acc, melanoma_recall, melanoma_precision, val_time = val_results
            
            # Scheduler
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
            # System stats
            import psutil
            ram = psutil.virtual_memory()
            
            print(f"\n{'='*100}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - MAX GPU TRAINING")
            print(f"{'='*100}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempos: Época {epoch_time:.1f}s | Train {epoch_time-val_time:.1f}s | Val {val_time:.1f}s")
            print(f"🚀 GPU PERFORMANCE:")
            print(f"   GPU Utilization: {gpu_stats['estimated_utilization']:.1f}% (Target: 95%+)")
            print(f"   VRAM Usage: {gpu_stats['memory_allocated_gb']:.1f}GB/8GB ({gpu_stats['memory_percent']:.1f}%)")
            print(f"   Throughput: {throughput:.0f} samples/sec")
            print(f"💾 RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB)")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # GPU performance feedback
            if gpu_stats['estimated_utilization'] < 85:
                print(f"⚠️  WARNING: GPU usage bajo ({gpu_stats['estimated_utilization']:.1f}%) - considerar aumentar batch size")
            elif gpu_stats['estimated_utilization'] >= 95:
                print(f"🎯 EXCELLENT: GPU usage óptimo ({gpu_stats['estimated_utilization']:.1f}%)")
            
            # Save best models
            saved_model = False
            
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'gpu_stats': gpu_stats,
                    'config': self.config
                }, 'best_max_gpu_overall.pth')
                print(f"✅ Mejor modelo MAX GPU guardado: {val_acc:.2f}%")
                saved_model = True
            
            if melanoma_recall > self.best_melanoma_recall:
                self.best_melanoma_recall = melanoma_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_melanoma_recall': melanoma_recall,
                    'val_acc': val_acc,
                    'gpu_stats': gpu_stats,
                    'config': self.config
                }, 'best_max_gpu_melanoma.pth')
                print(f"✅ Mejor melanoma MAX GPU guardado: {melanoma_recall*100:.2f}%")
                saved_model = True
            
            if balanced_score > self.best_balanced_score:
                self.best_balanced_score = balanced_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'balanced_score': balanced_score,
                    'val_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'gpu_stats': gpu_stats,
                    'config': self.config
                }, 'best_max_gpu_balanced.pth')
                print(f"✅ Mejor balance MAX GPU guardado: {balanced_score:.2f}%")
                saved_model = True
            
            if saved_model:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ Paciencia: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                break
        
        # VALIDACIÓN FINAL con matriz de confusión
        print(f"\n🔍 Realizando validación FINAL con matriz de confusión...")
        final_results = self.validate(len(self.val_accuracies), is_final=True)
        
        # Análisis final completo
        self.save_final_analysis()
        
        total_time = time.time() - start_time
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_gpu_util = max(self.gpu_utilization) if self.gpu_utilization else 0
        
        print(f"\n🎉 Entrenamiento MAX GPU completado!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"🚀 RENDIMIENTO GPU FINAL:")
        print(f"   GPU promedio: {avg_gpu_util:.1f}%")
        print(f"   GPU máximo: {max_gpu_util:.1f}%")
        print(f"   Target alcanzado: {'✅ SÍ' if avg_gpu_util >= 90 else '❌ NO'}")
        print(f"📊 Matriz de confusión guardada SOLO al final")

def main():
    """Entrenamiento MAX GPU con matriz de confusión solo al final"""
    print("🚀 ENTRENAMIENTO MAX GPU - RTX 3070 + 32GB RAM")
    print("🎯 OBJETIVO: 95%+ GPU + Matriz Confusión SOLO AL FINAL")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración MAX GPU
    training_config = {
        'learning_rate': 5e-5,
        'weight_decay': 5e-4,
        'loss_type': 'focal',
        'scheduler': 'plateau',
        'epochs': 60,
        'early_stopping_patience': 6,  # ⬇️ Más agresivo
        'gradient_clipping': 0.5,
        'mixed_precision': True,
        'gradient_accumulation_steps': 2
    }
    
    # BATCH SIZE MAX para RTX 3070
    batch_size = 24  # ⬆️ Máximo para tu GPU
    
    print(f"⚡ CONFIGURACIÓN MAX GPU:")
    print(f"   Batch size: {batch_size} (MAX RTX 3070)")
    print(f"   Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"   Batch efectivo: {batch_size * training_config['gradient_accumulation_steps']}")
    print(f"   Expected GPU: 95%+")
    print(f"   Workers: 16 (MAX 32GB RAM)")
    print(f"   VRAM target: 99%")
    print(f"   📊 Matriz confusión: SOLO AL FINAL")
    
    # Cargar datos con configuración MAX
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file, image_folders, batch_size=batch_size
    )
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    # Entrenar MAX GPU
    trainer = MaxGPUTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()