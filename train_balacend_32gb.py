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

# CONFIGURACIÓN BALANCEADA para evitar saturación RAM
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '8'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(8)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class BalancedRAMGPUTrainer:
    """Trainer balanceado con matriz de confusión completa"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # GPU optimizada pero RAM controlada
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)
        
        # AMP
        self.scaler = GradScaler()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Loss
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=self.class_weights)
        
        # Tracking con matrices de confusión
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.learning_rates = []
        self.confusion_matrices = []  # ⬆️ NUEVO
        self.classification_reports = []  # ⬆️ NUEVO
        
        # Best metrics
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 Trainer BALANCEADO con Análisis Completo inicializado:")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Workers: {train_loader.num_workers}")
        print(f"   Clases: {len(label_encoder.classes_)}")
        print(f"   Clases: {list(label_encoder.classes_)}")
        
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
        """Calcular métricas detalladas y matriz de confusión"""
        
        # Matriz de confusión
        cm = confusion_matrix(all_targets, all_predicted)
        
        # Classification report
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
    
    def plot_confusion_matrix(self, cm, epoch, save_path):
        """Crear gráfico de matriz de confusión"""
        plt.figure(figsize=(12, 10))
        
        # Normalizar matriz
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear heatmap
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
            cbar_kws={'label': 'Accuracy Normalizada'}
        )
        
        plt.title(f'Matriz de Confusión - CIFF-Net Balanceado\nÉpoca {epoch+1}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicción', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        
        # Highlight melanoma row/column
        melanoma_idx = None
        for i, class_name in enumerate(self.label_encoder.classes_):
            if 'mel' in class_name.lower():
                melanoma_idx = i
                break
        
        if melanoma_idx is not None:
            # Add red lines around melanoma
            plt.axhline(y=melanoma_idx, color='red', linewidth=3, alpha=0.7)
            plt.axhline(y=melanoma_idx+1, color='red', linewidth=3, alpha=0.7)
            plt.axvline(x=melanoma_idx, color='red', linewidth=3, alpha=0.7)
            plt.axvline(x=melanoma_idx+1, color='red', linewidth=3, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, epoch, cm, class_metrics, melanoma_recall, melanoma_precision):
        """Crear reporte detallado del modelo"""
        
        report_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REPORTE DETALLADO CIFF-NET BALANCEADO                    ║
║                                ÉPOCA {epoch+1:2d}                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 MÉTRICAS GENERALES:
   • Accuracy General: {self.val_accuracies[-1]:.2f}%
   • Loss de Validación: {self.val_losses[-1]:.4f}
   • Learning Rate: {self.learning_rates[-1]:.2e}

🩺 MÉTRICAS MELANOMA (CRÍTICAS):
   • Recall (Sensibilidad): {melanoma_recall*100:.2f}%
   • Precision: {melanoma_precision*100:.2f}%
   • F1-Score: {2*(melanoma_recall*melanoma_precision)/(melanoma_recall+melanoma_precision)*100 if (melanoma_recall+melanoma_precision) > 0 else 0:.2f}%

📊 MÉTRICAS POR CLASE:
"""
        
        for class_name, metrics in class_metrics.items():
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
        
        # Análisis de matriz de confusión
        report_text += f"""
🔍 ANÁLISIS MATRIZ DE CONFUSIÓN:
   • Diagonal principal (aciertos): {np.trace(cm)} / {np.sum(cm)}
   • Accuracy calculada: {np.trace(cm) / np.sum(cm) * 100:.2f}%
   
📈 ERRORES MÁS COMUNES:
"""
        
        # Encontrar errores más comunes (off-diagonal)
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
        
        report_text += f"""
⚖️ BALANCE DEL MODELO:
   • Score Balanceado: {self.compute_balanced_score(self.val_accuracies[-1], melanoma_recall):.2f}%
   • Mejor Accuracy hasta ahora: {self.best_overall_acc:.2f}%
   • Mejor Melanoma Recall hasta ahora: {self.best_melanoma_recall*100:.2f}%

💾 RECURSOS DEL SISTEMA:
"""
        
        import psutil
        ram = psutil.virtual_memory()
        gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        report_text += f"   • RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB)\n"
        report_text += f"   • GPU: {(gpu_memory/8)*100:.0f}% ({gpu_memory:.1f}GB/8GB)\n"
        
        return report_text
    
    def compute_balanced_score(self, accuracy, melanoma_recall):
        """Score balanceado que prioriza melanoma"""
        return 0.6 * accuracy + 0.4 * (melanoma_recall * 100)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Gradient accumulation para simular batch más grande
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"🔥 BALANCED Epoch {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
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
            
            # Monitor resources
            import psutil
            gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            gpu_percent = (gpu_memory / 8.0) * 100
            ram_percent = psutil.virtual_memory().percent
            
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_percent:.0f}%",
                'RAM': f"{ram_percent:.0f}%",
                'VRAM': f"{gpu_memory:.1f}GB"
            })
            
            if batch_idx % 20 == 0:
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
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, accuracy
    
    def validate(self, epoch):
        """Validación con análisis completo"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating BALANCED"):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Probabilidades para análisis adicional
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calcular métricas detalladas
        cm, report, class_metrics = self.compute_detailed_metrics(all_predicted, all_targets)
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        # Guardar datos
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        self.confusion_matrices.append(cm)
        self.classification_reports.append(report)
        
        # Crear matriz de confusión visual
        cm_path = f'confusion_matrix_epoch_{epoch+1}.png'
        self.plot_confusion_matrix(cm, epoch, cm_path)
        
        # Crear reporte detallado
        detailed_report = self.create_comprehensive_report(epoch, cm, class_metrics, melanoma_recall, melanoma_precision)
        
        # Guardar reporte
        with open(f'detailed_report_epoch_{epoch+1}.txt', 'w', encoding='utf-8') as f:
            f.write(detailed_report)
        
        # Mostrar reporte en consola
        print(detailed_report)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision, cm, class_metrics
    
    def save_final_analysis(self):
        """Guardar análisis final completo"""
        
        # Gráficos de entrenamiento
        plt.figure(figsize=(20, 15))
        
        # Loss
        plt.subplot(3, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.title('Pérdida Durante Entrenamiento', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(3, 3, 2)
        plt.plot(self.train_accuracies, label='Train Acc', linewidth=2)
        plt.plot(self.val_accuracies, label='Val Acc', linewidth=2)
        plt.title('Accuracy Durante Entrenamiento', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma Recall
        plt.subplot(3, 3, 3)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=2)
        plt.title('Melanoma Recall (Crítico)', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(3, 3, 4)
        plt.plot(self.learning_rates, color='green', linewidth=2)
        plt.title('Learning Rate', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Train-Val Gap
        plt.subplot(3, 3, 5)
        if len(self.train_accuracies) == len(self.val_accuracies):
            gap = [train - val for train, val in zip(self.train_accuracies, self.val_accuracies)]
            plt.plot(gap, color='purple', linewidth=2)
            plt.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Warning (5%)')
            plt.axhline(y=10, color='red', linestyle='-', alpha=0.7, label='Overfitting (10%)')
            plt.title('Train-Val Gap (Overfitting Check)', fontweight='bold')
            plt.xlabel('Época')
            plt.ylabel('Gap (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Métricas Comparativas
        plt.subplot(3, 3, 6)
        epochs = range(1, len(self.val_accuracies) + 1)
        plt.plot(epochs, self.val_accuracies, label='Overall Acc', linewidth=2)
        plt.plot(epochs, self.melanoma_recalls, label='Melanoma Recall', linewidth=2)
        balanced_scores = [self.compute_balanced_score(acc, mel_rec/100) for acc, mel_rec in zip(self.val_accuracies, self.melanoma_recalls)]
        plt.plot(epochs, balanced_scores, label='Balanced Score', linewidth=2, linestyle='--')
        plt.title('Métricas Comparativas', fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Última matriz de confusión
        if self.confusion_matrices:
            plt.subplot(3, 3, 7)
            cm = self.confusion_matrices[-1]
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            plt.title('Matriz de Confusión Final', fontweight='bold')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
        
        plt.tight_layout()
        plt.savefig('analisis_completo_ciffnet_balanceado.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Análisis completo guardado en 'analisis_completo_ciffnet_balanceado.png'")
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO BALANCEADO CON ANÁLISIS COMPLETO...")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate con análisis completo
            val_loss, val_acc, melanoma_recall, melanoma_precision, cm, class_metrics = self.validate(epoch)
            
            # Scheduler
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
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
                    'confusion_matrix': cm,
                    'class_metrics': class_metrics,
                    'config': self.config
                }, 'best_balanced_overall_with_cm.pth')
                saved_model = True
            
            if melanoma_recall > self.best_melanoma_recall:
                self.best_melanoma_recall = melanoma_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_melanoma_recall': melanoma_recall,
                    'val_acc': val_acc,
                    'confusion_matrix': cm,
                    'class_metrics': class_metrics,
                    'config': self.config
                }, 'best_balanced_melanoma_with_cm.pth')
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
                    'confusion_matrix': cm,
                    'class_metrics': class_metrics,
                    'config': self.config
                }, 'best_balanced_score_with_cm.pth')
                saved_model = True
            
            if saved_model:
                patience_counter = 0
                print(f"✅ Modelo mejorado guardado con matriz de confusión!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                break
        
        # Análisis final
        self.save_final_analysis()
        
        total_time = time.time() - start_time
        print(f"\n🎉 Entrenamiento BALANCEADO completado!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"📊 Matrices de confusión guardadas para cada época")
        print(f"📋 Reportes detallados guardados para cada época")

def main():
    """Entrenamiento balanceado con análisis completo"""
    print("🚀 ENTRENAMIENTO BALANCEADO CON MATRIZ DE CONFUSIÓN")
    print("🎯 ANÁLISIS COMPLETO DEL MODELO")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración BALANCEADA
    training_config = {
        'learning_rate': 5e-5,
        'weight_decay': 5e-4,
        'loss_type': 'focal',
        'scheduler': 'plateau',
        'epochs': 60,
        'early_stopping_patience': 10,
        'gradient_clipping': 0.5,
        'mixed_precision': True,
        'gradient_accumulation_steps': 2
    }
    
    batch_size = 16
    
    print(f"📊 ANÁLISIS INCLUIDO:")
    print(f"   ✅ Matriz de confusión por época")
    print(f"   ✅ Reporte detallado por clase")
    print(f"   ✅ Métricas de melanoma específicas")
    print(f"   ✅ Análisis de errores comunes")
    print(f"   ✅ Tracking de overfitting")
    print(f"   ✅ Score balanceado")
    
    # Cargar datos
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
    
    # Entrenar con análisis completo
    trainer = BalancedRAMGPUTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()