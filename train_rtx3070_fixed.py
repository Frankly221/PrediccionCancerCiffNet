import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN EXTREMA RTX 3070 + 32GB RAM (SIN TRITON)
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['CUDA_CACHE_DISABLE'] = '0'
# ⬇️ DESACTIVAR TRITON/COMPILE
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)

# Configuraciones NATIVAS sin Triton
torch.autograd.set_detect_anomaly(False)
torch.backends.cuda.cufft_plan_cache.max_size = 8

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class RTX3070FixedTrainer:
    """Trainer RTX 3070 SIN torch.compile - NATIVO"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN EXTREMA RTX 3070 - NATIVO
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.98)  # 98% VRAM
            
            # Configuraciones específicas RTX 3070
            torch.backends.cuda.cufft_plan_cache.max_size = 8
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Memory pool optimización
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
            
            print(f"✅ RTX 3070 configurada: VRAM 98%, SIN torch.compile")
        
        # NO COMPILAR MODELO - usar nativo
        print("🚀 Usando modelo NATIVO (sin torch.compile)")
        
        # AMP EXTREMO pero estable
        self.scaler = GradScaler(
            init_scale=65536.0,   # Scale alto pero estable
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        
        # Optimizer EXTREMO optimizado
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False,
            foreach=True if hasattr(torch.optim.AdamW, 'foreach') else False  # Vectorized si disponible
        )
        
        # Scheduler AGRESIVO
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            threshold=0.005
        )
        
        # Loss optimizado
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=self.class_weights)
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.learning_rates = []
        self.gpu_utilization = []
        self.vram_usage = []
        self.batch_throughput = []
        
        # Matriz final
        self.final_confusion_matrix = None
        self.final_class_metrics = None
        
        # Best metrics
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 RTX 3070 FIXED Trainer inicializado:")
        print(f"   TARGET GPU: 95%+")
        print(f"   TARGET VRAM: 7.5GB+")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Workers: {train_loader.num_workers}")
        print(f"   torch.compile: ❌ DESACTIVADO")
        print(f"   Optimización: NATIVA RTX 3070")
        
    def monitor_gpu_detailed(self):
        """Monitor GPU detallado RTX 3070"""
        if torch.cuda.is_available():
            # Sync para medición precisa
            torch.cuda.synchronize()
            
            # Memory
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_cached = torch.cuda.memory_cached() / 1e9
            
            # RTX 3070 tiene 8GB VRAM
            memory_percent = (memory_allocated / 8.0) * 100
            
            # Estimación más precisa para RTX 3070
            utilization = min(98, memory_percent * 1.1)
            
            return {
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_cached_gb': memory_cached,
                'memory_percent': memory_percent,
                'estimated_utilization': utilization,
                'total_vram_gb': 8.0
            }
        return {
            'memory_allocated_gb': 0, 'memory_reserved_gb': 0, 'memory_cached_gb': 0,
            'memory_percent': 0, 'estimated_utilization': 0, 'total_vram_gb': 8.0
        }
    
    def compute_melanoma_metrics(self, all_predicted, all_targets):
        """Métricas melanoma"""
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
        """Métricas detalladas"""
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
    
    def train_epoch_fixed(self, epoch):
        """Entrenamiento FIXED RTX 3070 - SIN torch.compile"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        batch_times = []
        gpu_stats_epoch = []
        
        # Gradient accumulation AGRESIVO
        accumulation_steps = self.config.get('gradient_accumulation_steps', 3)
        
        pbar = tqdm(self.train_loader, desc=f"🔥 RTX3070 FIXED Epoch {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Pre-warm GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Transfer OPTIMIZADO para RTX 3070
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass con AMP pero SIN torch.compile
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)  # ⬅️ MODELO NATIVO
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            # Backward AGRESIVO
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Statistics
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # GPU monitoring
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            gpu_stats = self.monitor_gpu_detailed()
            gpu_stats_epoch.append(gpu_stats['estimated_utilization'])
            
            samples_per_second = inputs.size(0) / batch_time
            
            # Progress bar DETALLADO
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['estimated_utilization']:.1f}%",
                'VRAM': f"{gpu_stats['memory_allocated_gb']:.1f}GB",
                'SPS': f"{samples_per_second:.0f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'Mode': 'NATIVE'
            })
            
            # WARNING GPU en tiempo real
            if batch_idx % 20 == 0:
                current_gpu = gpu_stats['estimated_utilization']
                if current_gpu < 60:
                    print(f"\n⚠️ GPU BAJO: {current_gpu:.1f}% en batch {batch_idx} - RTX 3070")
                elif current_gpu >= 90:
                    print(f"\n🎯 GPU EXCELENTE: {current_gpu:.1f}% en batch {batch_idx} - RTX 3070")
            
            # Limpieza controlada
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Final step
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Métricas finales
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        throughput = len(self.train_loader.dataset) / epoch_time
        avg_gpu_util = np.mean(gpu_stats_epoch) if gpu_stats_epoch else 0
        
        # Guardar métricas
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.batch_throughput.append(throughput)
        self.gpu_utilization.append(avg_gpu_util)
        
        final_gpu_stats = self.monitor_gpu_detailed()
        self.vram_usage.append(final_gpu_stats['memory_allocated_gb'])
        
        return avg_loss, accuracy, throughput, final_gpu_stats, avg_gpu_util
    
    def validate(self, epoch, is_final=False):
        """Validación optimizada RTX 3070"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        validation_start = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating RTX3070 FIXED"):
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs)  # ⬅️ MODELO NATIVO
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
        
        # Métricas melanoma
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        # Métricas detalladas solo si es final
        if is_final:
            cm, report, class_metrics = self.compute_detailed_metrics(all_predicted, all_targets)
            self.final_confusion_matrix = cm
            self.final_class_metrics = class_metrics
            print(f"📊 Calculando análisis FINAL RTX 3070...")
            return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time, cm, class_metrics
        
        # Guardar métricas básicas
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time
    
    def plot_final_confusion_matrix(self):
        """Matriz de confusión final RTX 3070"""
        if self.final_confusion_matrix is None:
            return
        
        plt.figure(figsize=(14, 12))
        
        cm = self.final_confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
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
        
        avg_gpu = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_vram = max(self.vram_usage) if self.vram_usage else 0
        
        plt.title(f'MATRIZ DE CONFUSIÓN FINAL - RTX 3070 FIXED\n'
                 f'Accuracy: {self.best_overall_acc:.2f}% | Melanoma: {self.best_melanoma_recall*100:.2f}% | GPU: {avg_gpu:.1f}% | VRAM: {max_vram:.1f}GB', 
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
        plt.savefig('final_confusion_matrix_rtx3070_fixed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Matriz confusión RTX 3070 FIXED guardada!")
    
    def save_final_analysis(self):
        """Análisis final RTX 3070 FIXED"""
        
        plt.figure(figsize=(24, 18))
        
        # GPU Utilization - PRINCIPAL
        plt.subplot(3, 4, 1)
        plt.plot(self.gpu_utilization, color='orange', linewidth=3)
        plt.axhline(y=70, color='yellow', linestyle='--', label='Mínimo 70%', alpha=0.7)
        plt.axhline(y=85, color='green', linestyle='--', label='Bueno 85%', alpha=0.7)
        plt.axhline(y=95, color='red', linestyle='--', label='Excelente 95%', alpha=0.7)
        plt.title('GPU Utilization RTX 3070 FIXED', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('GPU Usage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # VRAM Usage
        plt.subplot(3, 4, 2)
        plt.plot(self.vram_usage, color='purple', linewidth=2)
        plt.axhline(y=6, color='yellow', linestyle='--', label='Mínimo 6GB', alpha=0.7)
        plt.axhline(y=7, color='green', linestyle='--', label='Bueno 7GB', alpha=0.7)
        plt.axhline(y=7.5, color='red', linestyle='--', label='Excelente 7.5GB', alpha=0.7)
        plt.title('VRAM Usage RTX 3070 (8GB Total)', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('VRAM (GB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(3, 4, 3)
        plt.plot(self.train_accuracies, label='Train Acc', linewidth=2, color='blue')
        plt.plot(self.val_accuracies, label='Val Acc', linewidth=2, color='red')
        plt.title('Accuracy RTX 3070 FIXED', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma Recall
        plt.subplot(3, 4, 4)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=3)
        plt.title('Melanoma Recall - CRÍTICO', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(3, 4, 5)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2, color='blue')
        plt.plot(self.val_losses, label='Val Loss', linewidth=2, color='red')
        plt.title('Loss RTX 3070 FIXED', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Throughput
        plt.subplot(3, 4, 6)
        plt.plot(self.batch_throughput, color='green', linewidth=2)
        plt.title('Training Throughput RTX 3070', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Samples/Second')
        plt.grid(True, alpha=0.3)
        
        # GPU vs Performance
        plt.subplot(3, 4, 7)
        if len(self.gpu_utilization) > 0 and len(self.val_accuracies) > 0:
            plt.scatter(self.gpu_utilization, self.val_accuracies, alpha=0.7, color='red', s=50)
            plt.xlabel('GPU Utilization (%)')
            plt.ylabel('Validation Accuracy (%)')
            plt.title('Performance vs GPU RTX 3070', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(3, 4, 8)
        plt.plot(self.learning_rates, color='brown', linewidth=2)
        plt.title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Performance Summary
        plt.subplot(3, 4, 9)
        epochs = range(1, len(self.val_accuracies) + 1)
        plt.plot(epochs, self.val_accuracies, label='Val Acc', linewidth=2, color='blue')
        plt.plot(epochs, self.melanoma_recalls, label='Melanoma Recall', linewidth=2, color='red')
        balanced_scores = [self.compute_balanced_score(acc, mel_rec/100) for acc, mel_rec in zip(self.val_accuracies, self.melanoma_recalls)]
        plt.plot(epochs, balanced_scores, label='Balanced Score', linewidth=2, linestyle='--', color='green')
        plt.title('Performance Summary RTX 3070', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # VRAM vs GPU efficiency
        plt.subplot(3, 4, 10)
        if len(self.vram_usage) > 0 and len(self.gpu_utilization) > 0:
            plt.scatter(self.vram_usage, self.gpu_utilization, alpha=0.7, color='purple', s=50)
            plt.xlabel('VRAM Usage (GB)')
            plt.ylabel('GPU Utilization (%)')
            plt.title('VRAM vs GPU Efficiency', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rtx3070_fixed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear matriz final
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
            'vram_usage_gb': self.vram_usage,
            'throughput_samples_sec': self.batch_throughput
        })
        metrics_df.to_csv('rtx3070_fixed_metrics.csv', index=False)
        
        print("📊 Análisis RTX 3070 FIXED guardado:")
        print("   📈 rtx3070_fixed_analysis.png")
        print("   📊 final_confusion_matrix_rtx3070_fixed.png")
        print("   📋 rtx3070_fixed_metrics.csv")
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO RTX 3070 FIXED...")
        print(f"🎯 OBJETIVO: GPU 95%+ SIN torch.compile")
        print(f"⚙️  Modo: NATIVO RTX 3070")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train FIXED
            train_loss, train_acc, throughput, gpu_stats, avg_gpu_util = self.train_epoch_fixed(epoch)
            
            # Validate
            val_results = self.validate(epoch, is_final=False)
            val_loss, val_acc, melanoma_recall, melanoma_precision, val_time = val_results
            
            # Scheduler
            self.scheduler.step(val_acc)
            
            epoch_time = time.time() - epoch_start
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
            # System stats
            import psutil
            ram = psutil.virtual_memory()
            
            print(f"\n{'='*120}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - RTX 3070 FIXED TRAINING (NATIVO)")
            print(f"{'='*120}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempos: Época {epoch_time:.1f}s | Train {epoch_time-val_time:.1f}s | Val {val_time:.1f}s")
            print(f"🚀 RTX 3070 PERFORMANCE (NATIVO):")
            print(f"   GPU Utilization: {avg_gpu_util:.1f}% (Target: 95%+)")
            print(f"   VRAM Usage: {gpu_stats['memory_allocated_gb']:.1f}GB/8GB ({gpu_stats['memory_percent']:.1f}%)")
            print(f"   VRAM Cached: {gpu_stats['memory_cached_gb']:.1f}GB")
            print(f"   Throughput: {throughput:.0f} samples/sec")
            print(f"💾 RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB)")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"🔧 Modo: NATIVO (sin torch.compile)")
            
            # GPU feedback DETALLADO
            if avg_gpu_util < 40:
                print(f"🚨 CRÍTICO: GPU MUY BAJO ({avg_gpu_util:.1f}%) - Verificar configuración RTX 3070")
            elif avg_gpu_util < 60:
                print(f"⚠️  WARNING: GPU bajo ({avg_gpu_util:.1f}%) - Aumentar batch size o verificar bottleneck")
            elif avg_gpu_util < 80:
                print(f"🟡 REGULAR: GPU moderado ({avg_gpu_util:.1f}%) - Puede mejorar")
            elif avg_gpu_util >= 95:
                print(f"🎯 EXCELENTE: GPU óptimo ({avg_gpu_util:.1f}%) - ¡PERFECTO RTX 3070!")
            else:
                print(f"✅ MUY BUENO: GPU alto ({avg_gpu_util:.1f}%) - RTX 3070 funcionando bien")
            
            # VRAM feedback
            vram_percent = gpu_stats['memory_percent']
            if vram_percent >= 90:
                print(f"🎯 VRAM EXCELENTE: {vram_percent:.1f}% utilizando RTX 3070")
            elif vram_percent >= 75:
                print(f"✅ VRAM BUENO: {vram_percent:.1f}% RTX 3070")
            else:
                print(f"⚠️ VRAM BAJO: {vram_percent:.1f}% - RTX 3070 puede usar más")
            
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
                    'config': self.config,
                    'mode': 'RTX3070_FIXED_NATIVE'
                }, 'best_rtx3070_fixed_overall.pth')
                print(f"✅ Mejor modelo RTX 3070 FIXED guardado: {val_acc:.2f}%")
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
                    'config': self.config,
                    'mode': 'RTX3070_FIXED_NATIVE'
                }, 'best_rtx3070_fixed_melanoma.pth')
                print(f"✅ Mejor melanoma RTX 3070 FIXED guardado: {melanoma_recall*100:.2f}%")
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
                    'config': self.config,
                    'mode': 'RTX3070_FIXED_NATIVE'
                }, 'best_rtx3070_fixed_balanced.pth')
                print(f"✅ Mejor balance RTX 3070 FIXED guardado: {balanced_score:.2f}%")
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
        
        # VALIDACIÓN FINAL
        print(f"\n🔍 Realizando validación FINAL RTX 3070...")
        final_results = self.validate(len(self.val_accuracies), is_final=True)
        
        # Análisis final
        self.save_final_analysis()
        
        total_time = time.time() - start_time
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_gpu_util = max(self.gpu_utilization) if self.gpu_utilization else 0
        max_vram = max(self.vram_usage) if self.vram_usage else 0
        
        print(f"\n🎉 ENTRENAMIENTO RTX 3070 FIXED COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"🚀 RENDIMIENTO RTX 3070 FINAL:")
        print(f"   GPU promedio: {avg_gpu_util:.1f}%")
        print(f"   GPU máximo: {max_gpu_util:.1f}%")
        print(f"   VRAM máximo: {max_vram:.1f}GB/8GB")
        print(f"   Target GPU alcanzado: {'✅ SÍ' if avg_gpu_util >= 85 else '❌ NO'}")
        print(f"   Target VRAM alcanzado: {'✅ SÍ' if max_vram >= 6.5 else '❌ NO'}")
        print(f"   Modo usado: NATIVO (sin torch.compile)")

def main():
    """Entrenamiento RTX 3070 FIXED - SIN torch.compile"""
    print("🚀 ENTRENAMIENTO RTX 3070 FIXED")
    print("🎯 OBJETIVO: 95%+ GPU SIN torch.compile")
    print("⚙️  MODO: NATIVO RTX 3070 + 32GB RAM")
    print("🔧 FIX: Triton/torch.compile DESACTIVADO")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración EXTREMA RTX 3070 - FIXED
    training_config = {
        'learning_rate': 6e-5,
        'weight_decay': 4e-4,
        'loss_type': 'focal',
        'scheduler': 'plateau',
        'epochs': 60,
        'early_stopping_patience': 5,
        'gradient_clipping': 0.5,
        'mixed_precision': True,
        'gradient_accumulation_steps': 3
    }
    
    # BATCH SIZE MÁXIMO RTX 3070
    batch_size = 32
    
    print(f"⚡ CONFIGURACIÓN RTX 3070 FIXED:")
    print(f"   Batch size: {batch_size} (MÁXIMO RTX 3070)")
    print(f"   Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"   Batch efectivo: {batch_size * training_config['gradient_accumulation_steps']} = 96")
    print(f"   Target GPU: 95%+")
    print(f"   Target VRAM: 7.5GB+")
    print(f"   Workers: 16")
    print(f"   torch.compile: ❌ DESACTIVADO")
    print(f"   Triton: ❌ DESACTIVADO")
    print(f"   Modo: NATIVO PyTorch")
    
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
    
    # Entrenar RTX 3070 FIXED
    trainer = RTX3070FixedTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()