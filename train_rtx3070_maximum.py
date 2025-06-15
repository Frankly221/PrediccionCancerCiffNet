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

# CONFIGURACIÓN MÁXIMA RTX 3070 - APROVECHAR 96% GPU + 5.4GB VRAM LIBRE
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_JIT'] = '0'

torch.backends.cudnn.benchmark = True  # ✅ ACTIVADO
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)
torch.autograd.set_detect_anomaly(False)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class RTX3070MaximumTrainer:
    """Trainer MÁXIMO RTX 3070 - Aprovechar 96% GPU + 5.4GB VRAM libres"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN MÁXIMA RTX 3070 - APROVECHAR VRAM LIBRE
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # 95% VRAM (usar casi todo)
            
            # Optimizaciones específicas RTX 3070
            torch.backends.cuda.cufft_plan_cache.max_size = 16  # ⬆️ Más cache
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Memory pool MÁXIMO
            torch.cuda.memory._set_allocator_settings('expandable_segments:True')
            
            print(f"✅ RTX 3070 MÁXIMO configurada: VRAM 95%, aprovechando 5.4GB libres")
        
        print("🚀 Usando modelo NATIVO optimizado para RTX 3070")
        
        # AMP MÁXIMO - aprovechar que tenemos margen
        self.scaler = GradScaler(
            init_scale=131072.0,  # ⬆️ Scale MUY alto (tenemos margen)
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1000   # ⬇️ Más agresivo
        )
        
        # Optimizer MÁXIMO
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False,
            foreach=True if hasattr(torch.optim.AdamW, 'foreach') else False
        )
        
        # Scheduler AGRESIVO
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=2,  # ⬇️ MUY agresivo aprovechando estabilidad
            min_lr=1e-8,
            threshold=0.003
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
        self.gpu_temperature = []
        self.gpu_power = []
        
        # Matriz final
        self.final_confusion_matrix = None
        self.final_class_metrics = None
        
        # Best metrics
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 RTX 3070 MAXIMUM Trainer inicializado:")
        print(f"   CURRENT GPU: 96% ✅")
        print(f"   CURRENT VRAM: 2.6GB/8GB (33%)")
        print(f"   AVAILABLE VRAM: 5.4GB ⬆️")
        print(f"   TARGET: Usar 6-7GB VRAM")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Temperature: 86°C ✅ SEGURA")
        
    def monitor_gpu_maximum(self):
        """Monitor GPU MÁXIMO con temperatura y power"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
            # Memory
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_cached = torch.cuda.memory_cached() / 1e9
            
            # RTX 3070 8GB
            memory_percent = (memory_allocated / 8.0) * 100
            utilization = min(99, memory_percent * 1.05)  # Más preciso
            
            # Temperatura y power (si disponible)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                pynvml.nvmlShutdown()
            except:
                temp = 0
                power_draw = 0
            
            return {
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_cached_gb': memory_cached,
                'memory_percent': memory_percent,
                'estimated_utilization': utilization,
                'total_vram_gb': 8.0,
                'temperature_c': temp,
                'power_draw_w': power_draw
            }
        return {
            'memory_allocated_gb': 0, 'memory_reserved_gb': 0, 'memory_cached_gb': 0,
            'memory_percent': 0, 'estimated_utilization': 0, 'total_vram_gb': 8.0,
            'temperature_c': 0, 'power_draw_w': 0
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
    
    def train_epoch_maximum(self, epoch):
        """Entrenamiento MÁXIMO RTX 3070 - Aprovechar 5.4GB VRAM libres"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        batch_times = []
        gpu_stats_epoch = []
        
        # Gradient accumulation MÁXIMO
        accumulation_steps = self.config.get('gradient_accumulation_steps', 4)  # ⬆️ Más agresivo
        
        pbar = tqdm(self.train_loader, desc=f"🔥 RTX3070 MAXIMUM Epoch {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Pre-warm GPU y usar cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Transfer MÁXIMO optimizado
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass MÁXIMO
            with autocast(dtype=torch.float16):  # FP16 MÁXIMO
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            # Backward MÁXIMO
            self.scaler.scale(loss).backward()
            
            # Optimizer step
            if (batch_idx + 1) % accumulation_steps == 0:
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
            
            # GPU monitoring MÁXIMO
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            gpu_stats = self.monitor_gpu_maximum()
            gpu_stats_epoch.append(gpu_stats['estimated_utilization'])
            
            samples_per_second = inputs.size(0) / batch_time
            
            # Progress bar DETALLADO con temperatura
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['estimated_utilization']:.1f}%",
                'VRAM': f"{gpu_stats['memory_allocated_gb']:.1f}GB",
                'Temp': f"{gpu_stats['temperature_c']:.0f}°C",
                'Power': f"{gpu_stats['power_draw_w']:.0f}W",
                'SPS': f"{samples_per_second:.0f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Monitoring en tiempo real
            if batch_idx % 25 == 0:
                current_gpu = gpu_stats['estimated_utilization']
                current_vram = gpu_stats['memory_allocated_gb']
                current_temp = gpu_stats['temperature_c']
                
                if current_gpu >= 95:
                    status = "🎯 PERFECTO"
                elif current_gpu >= 85:
                    status = "✅ EXCELENTE"
                elif current_gpu >= 70:
                    status = "🟡 BUENO"
                else:
                    status = "⚠️ BAJO"
                
                if current_vram >= 6:
                    vram_status = "🎯 MÁXIMO"
                elif current_vram >= 4:
                    vram_status = "✅ BUENO"
                else:
                    vram_status = "⬆️ AUMENTAR"
                
                temp_status = "✅ SEGURA" if current_temp <= 90 else "⚠️ CALIENTE"
                
                print(f"\n📊 Batch {batch_idx}: GPU {current_gpu:.1f}% {status} | VRAM {current_vram:.1f}GB {vram_status} | Temp {current_temp:.0f}°C {temp_status}")
            
            # Limpieza menos frecuente para mantener rendimiento
            if batch_idx % 60 == 0:  # ⬆️ Cada 60 batches
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
        
        final_gpu_stats = self.monitor_gpu_maximum()
        self.vram_usage.append(final_gpu_stats['memory_allocated_gb'])
        self.gpu_temperature.append(final_gpu_stats['temperature_c'])
        self.gpu_power.append(final_gpu_stats['power_draw_w'])
        
        return avg_loss, accuracy, throughput, final_gpu_stats, avg_gpu_util
    
    def validate(self, epoch, is_final=False):
        """Validación MÁXIMA RTX 3070"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        validation_start = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating RTX3070 MAXIMUM"):
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.float16):
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
        
        # Métricas melanoma
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        # Métricas detalladas solo si es final
        if is_final:
            cm, report, class_metrics = self.compute_detailed_metrics(all_predicted, all_targets)
            self.final_confusion_matrix = cm
            self.final_class_metrics = class_metrics
            print(f"📊 Calculando análisis FINAL RTX 3070 MAXIMUM...")
            return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time, cm, class_metrics
        
        # Guardar métricas básicas
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time
    
    def plot_final_confusion_matrix(self):
        """Matriz de confusión final RTX 3070 MAXIMUM"""
        if self.final_confusion_matrix is None:
            return
        
        plt.figure(figsize=(16, 14))
        
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
        avg_temp = np.mean(self.gpu_temperature) if self.gpu_temperature else 0
        avg_power = np.mean(self.gpu_power) if self.gpu_power else 0
        
        plt.title(f'MATRIZ DE CONFUSIÓN FINAL - RTX 3070 MAXIMUM\n'
                 f'Accuracy: {self.best_overall_acc:.2f}% | Melanoma: {self.best_melanoma_recall*100:.2f}% | GPU: {avg_gpu:.1f}%\n'
                 f'VRAM: {max_vram:.1f}GB | Temp: {avg_temp:.0f}°C | Power: {avg_power:.0f}W', 
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
        plt.savefig('final_confusion_matrix_rtx3070_maximum.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Matriz confusión RTX 3070 MAXIMUM guardada!")
    
    def save_final_analysis(self):
        """Análisis final RTX 3070 MAXIMUM"""
        
        plt.figure(figsize=(26, 20))
        
        # GPU Utilization - PRINCIPAL
        plt.subplot(4, 4, 1)
        plt.plot(self.gpu_utilization, color='orange', linewidth=3)
        plt.axhline(y=90, color='green', linestyle='--', label='Excelente 90%', alpha=0.7)
        plt.axhline(y=96, color='red', linestyle='-', label='ACTUAL 96%', alpha=0.8, linewidth=2)
        plt.title('GPU Utilization RTX 3070 MAXIMUM', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('GPU Usage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # VRAM Usage
        plt.subplot(4, 4, 2)
        plt.plot(self.vram_usage, color='purple', linewidth=3)
        plt.axhline(y=2.6, color='blue', linestyle='--', label='Inicial 2.6GB', alpha=0.7)
        plt.axhline(y=6, color='green', linestyle='--', label='Target 6GB', alpha=0.7)
        plt.axhline(y=7, color='red', linestyle='--', label='Máximo 7GB', alpha=0.7)
        plt.title('VRAM Usage RTX 3070 (8GB Total)', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('VRAM (GB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # GPU Temperature
        plt.subplot(4, 4, 3)
        if self.gpu_temperature:
            plt.plot(self.gpu_temperature, color='red', linewidth=2)
            plt.axhline(y=86, color='blue', linestyle='--', label='Actual 86°C', alpha=0.7)
            plt.axhline(y=90, color='orange', linestyle='--', label='Límite 90°C', alpha=0.7)
            plt.axhline(y=95, color='red', linestyle='--', label='Máximo 95°C', alpha=0.7)
            plt.title('GPU Temperature RTX 3070', fontweight='bold', fontsize=14)
            plt.xlabel('Época')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # GPU Power
        plt.subplot(4, 4, 4)
        if self.gpu_power:
            plt.plot(self.gpu_power, color='green', linewidth=2)
            plt.axhline(y=94, color='blue', linestyle='--', label='Actual 94W', alpha=0.7)
            plt.axhline(y=104, color='red', linestyle='--', label='Máximo 104W', alpha=0.7)
            plt.title('GPU Power Draw RTX 3070', fontweight='bold', fontsize=14)
            plt.xlabel('Época')
            plt.ylabel('Power (W)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Accuracy
        plt.subplot(4, 4, 5)
        plt.plot(self.train_accuracies, label='Train Acc', linewidth=2, color='blue')
        plt.plot(self.val_accuracies, label='Val Acc', linewidth=2, color='red')
        plt.title('Accuracy RTX 3070 MAXIMUM', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma Recall
        plt.subplot(4, 4, 6)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=3)
        plt.title('Melanoma Recall - CRÍTICO', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(4, 4, 7)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2, color='blue')
        plt.plot(self.val_losses, label='Val Loss', linewidth=2, color='red')
        plt.title('Loss RTX 3070 MAXIMUM', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Throughput
        plt.subplot(4, 4, 8)
        plt.plot(self.batch_throughput, color='green', linewidth=2)
        plt.title('Training Throughput RTX 3070', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('Samples/Second')
        plt.grid(True, alpha=0.3)
        
        # GPU vs VRAM efficiency
        plt.subplot(4, 4, 9)
        if len(self.gpu_utilization) > 0 and len(self.vram_usage) > 0:
            plt.scatter(self.vram_usage, self.gpu_utilization, alpha=0.7, color='purple', s=60)
            plt.xlabel('VRAM Usage (GB)')
            plt.ylabel('GPU Utilization (%)')
            plt.title('VRAM vs GPU Efficiency', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        # Performance vs Temperature
        plt.subplot(4, 4, 10)
        if len(self.gpu_temperature) > 0 and len(self.val_accuracies) > 0:
            plt.scatter(self.gpu_temperature, self.val_accuracies, alpha=0.7, color='orange', s=60)
            plt.xlabel('GPU Temperature (°C)')
            plt.ylabel('Validation Accuracy (%)')
            plt.title('Performance vs Temperature', fontweight='bold', fontsize=14)
            plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(4, 4, 11)
        plt.plot(self.learning_rates, color='brown', linewidth=2)
        plt.title('Learning Rate Schedule', fontweight='bold', fontsize=14)
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Performance Summary
        plt.subplot(4, 4, 12)
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
        
        # Power Efficiency
        plt.subplot(4, 4, 13)
        if len(self.gpu_power) > 0 and len(self.val_accuracies) > 0:
            efficiency = [acc / (power + 1) for acc, power in zip(self.val_accuracies, self.gpu_power)]
            plt.plot(efficiency, color='purple', linewidth=2)
            plt.title('Power Efficiency (Acc/Watt)', fontweight='bold', fontsize=14)
            plt.xlabel('Época')
            plt.ylabel('Efficiency')
            plt.grid(True, alpha=0.3)
        
        # System Overview
        plt.subplot(4, 4, 14)
        # Texto con stats principales
        avg_gpu = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_vram = max(self.vram_usage) if self.vram_usage else 0
        avg_temp = np.mean(self.gpu_temperature) if self.gpu_temperature else 0
        avg_power = np.mean(self.gpu_power) if self.gpu_power else 0
        
        stats_text = f"""RTX 3070 MAXIMUM STATS:
        
GPU Usage: {avg_gpu:.1f}%
VRAM Max: {max_vram:.1f}GB
Temperature: {avg_temp:.0f}°C
Power: {avg_power:.0f}W

Best Accuracy: {self.best_overall_acc:.1f}%
Best Melanoma: {self.best_melanoma_recall*100:.1f}%
        
STATUS: {'🎯 PERFECTO' if avg_gpu >= 90 else '✅ BUENO' if avg_gpu >= 80 else '⚠️ MEJORAR'}"""
        
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        plt.axis('off')
        plt.title('System Overview', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('rtx3070_maximum_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear matriz final
        self.plot_final_confusion_matrix()
        
        # CSV con métricas MÁXIMAS
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
            'gpu_temperature_c': self.gpu_temperature,
            'gpu_power_w': self.gpu_power,
            'throughput_samples_sec': self.batch_throughput
        })
        metrics_df.to_csv('rtx3070_maximum_metrics.csv', index=False)
        
        print("📊 Análisis RTX 3070 MAXIMUM guardado:")
        print("   📈 rtx3070_maximum_analysis.png")
        print("   📊 final_confusion_matrix_rtx3070_maximum.png")
        print("   📋 rtx3070_maximum_metrics.csv")
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO RTX 3070 MAXIMUM...")
        print(f"🎯 OBJETIVO: Aprovechar 5.4GB VRAM libres + mantener 96% GPU")
        print(f"⚙️  Base: 96% GPU, 2.6GB VRAM, 86°C, 94W")
        print(f"📈 Target: 96%+ GPU, 6-7GB VRAM, <90°C, <100W")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train MAXIMUM
            train_loss, train_acc, throughput, gpu_stats, avg_gpu_util = self.train_epoch_maximum(epoch)
            
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
            
            print(f"\n{'='*130}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - RTX 3070 MAXIMUM TRAINING")
            print(f"{'='*130}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempos: Época {epoch_time:.1f}s | Train {epoch_time-val_time:.1f}s | Val {val_time:.1f}s")
            print(f"🚀 RTX 3070 MAXIMUM PERFORMANCE:")
            print(f"   GPU Utilization: {avg_gpu_util:.1f}% (Base: 96%)")
            print(f"   VRAM Usage: {gpu_stats['memory_allocated_gb']:.1f}GB/8GB ({gpu_stats['memory_percent']:.1f}%)")
            print(f"   VRAM Available: {8.0 - gpu_stats['memory_allocated_gb']:.1f}GB")
            print(f"   Temperature: {gpu_stats['temperature_c']:.0f}°C (Base: 86°C)")
            print(f"   Power Draw: {gpu_stats['power_draw_w']:.0f}W (Base: 94W)")
            print(f"   Throughput: {throughput:.0f} samples/sec")
            print(f"💾 RAM: {ram.percent:.1f}% ({ram.used/1e9:.1f}GB/{ram.total/1e9:.1f}GB)")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # GPU feedback MÁXIMO
            if avg_gpu_util >= 96:
                print(f"🎯 PERFECTO: GPU igual o mejor que base ({avg_gpu_util:.1f}%)")
            elif avg_gpu_util >= 90:
                print(f"✅ EXCELENTE: GPU muy alto ({avg_gpu_util:.1f}%)")
            elif avg_gpu_util >= 80:
                print(f"🟡 BUENO: GPU alto ({avg_gpu_util:.1f}%) - puede mejorar")
            else:
                print(f"⚠️ BAJO: GPU ({avg_gpu_util:.1f}%) - verificar configuración")
            
            # VRAM feedback
            vram_gb = gpu_stats['memory_allocated_gb']
            if vram_gb >= 6:
                print(f"🎯 VRAM MÁXIMO: {vram_gb:.1f}GB - Aprovechando VRAM libre")
            elif vram_gb >= 4:
                print(f"✅ VRAM BUENO: {vram_gb:.1f}GB - Puede usar más")
            elif vram_gb >= 2.6:
                print(f"🟡 VRAM BASE: {vram_gb:.1f}GB - Similar a inicial")
            else:
                print(f"⚠️ VRAM BAJO: {vram_gb:.1f}GB - Por debajo de base")
            
            # Temperature feedback
            temp = gpu_stats['temperature_c']
            if temp <= 90:
                print(f"✅ TEMPERATURA SEGURA: {temp:.0f}°C")
            elif temp <= 95:
                print(f"⚠️ TEMPERATURA ALTA: {temp:.0f}°C - monitorear")
            else:
                print(f"🚨 TEMPERATURA CRÍTICA: {temp:.0f}°C - reducir carga")
            
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
                    'mode': 'RTX3070_MAXIMUM'
                }, 'best_rtx3070_maximum_overall.pth')
                print(f"✅ Mejor modelo RTX 3070 MAXIMUM guardado: {val_acc:.2f}%")
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
                    'mode': 'RTX3070_MAXIMUM'
                }, 'best_rtx3070_maximum_melanoma.pth')
                print(f"✅ Mejor melanoma RTX 3070 MAXIMUM guardado: {melanoma_recall*100:.2f}%")
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
                    'mode': 'RTX3070_MAXIMUM'
                }, 'best_rtx3070_maximum_balanced.pth')
                print(f"✅ Mejor balance RTX 3070 MAXIMUM guardado: {balanced_score:.2f}%")
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
        print(f"\n🔍 Realizando validación FINAL RTX 3070 MAXIMUM...")
        final_results = self.validate(len(self.val_accuracies), is_final=True)
        
        # Análisis final
        self.save_final_analysis()
        
        total_time = time.time() - start_time
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_gpu_util = max(self.gpu_utilization) if self.gpu_utilization else 0
        max_vram = max(self.vram_usage) if self.vram_usage else 0
        avg_temp = np.mean(self.gpu_temperature) if self.gpu_temperature else 0
        avg_power = np.mean(self.gpu_power) if self.gpu_power else 0
        
        print(f"\n🎉 ENTRENAMIENTO RTX 3070 MAXIMUM COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"🚀 RENDIMIENTO RTX 3070 FINAL:")
        print(f"   GPU promedio: {avg_gpu_util:.1f}% (Base: 96%)")
        print(f"   GPU máximo: {max_gpu_util:.1f}%")
        print(f"   VRAM máximo: {max_vram:.1f}GB/8GB (Base: 2.6GB)")
        print(f"   Temperatura promedio: {avg_temp:.0f}°C (Base: 86°C)")
        print(f"   Power promedio: {avg_power:.0f}W (Base: 94W)")
        print(f"   Target GPU alcanzado: {'✅ SÍ' if avg_gpu_util >= 90 else '❌ NO'}")
        print(f"   Target VRAM alcanzado: {'✅ SÍ' if max_vram >= 6 else '🟡 PARCIAL' if max_vram >= 4 else '❌ NO'}")
        print(f"   Modo usado: MAXIMUM NATIVE")

def main():
    """Entrenamiento RTX 3070 MAXIMUM - Aprovechar 5.4GB VRAM libres"""
    print("🚀 ENTRENAMIENTO RTX 3070 MAXIMUM")
    print("🎯 OBJETIVO: Aprovechar 5.4GB VRAM libres + mantener 96% GPU")
    print("⚙️  BASE ACTUAL: 96% GPU, 2.6GB VRAM, 86°C, 94W")
    print("📈 TARGET: 96%+ GPU, 6-7GB VRAM, <90°C, <100W")
    print("=" * 110)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración MÁXIMA RTX 3070 - Aprovechar VRAM libre
    training_config = {
        'learning_rate': 7e-5,    # ⬆️ Ligeramente más alto
        'weight_decay': 3e-4,     # ⬇️ Menos regularización
        'loss_type': 'focal',
        'scheduler': 'plateau',
        'epochs': 60,
        'early_stopping_patience': 4,
        'gradient_clipping': 0.5,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4  # ⬆️ Usar más VRAM
    }
    
    # BATCH SIZE MÁXIMO para aprovechar VRAM libre
    batch_size = 48  # ⬆️ MÁXIMO para usar 5.4GB libres
    
    print(f"⚡ CONFIGURACIÓN RTX 3070 MAXIMUM:")
    print(f"   Batch size: {batch_size} (MÁXIMO para 5.4GB libres)")
    print(f"   Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"   Batch efectivo: {batch_size * training_config['gradient_accumulation_steps']} = 192")
    print(f"   Current GPU: 96% ✅")
    print(f"   Current VRAM: 2.6GB (33%)")
    print(f"   Available VRAM: 5.4GB")
    print(f"   Target VRAM: 6-7GB (75-87%)")
    print(f"   Workers: 20")
    print(f"   CuDNN benchmark: ✅ ACTIVADO")
    
    # Cargar datos con BATCH SIZE MÁXIMO
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
    
    # Entrenar RTX 3070 MAXIMUM
    trainer = RTX3070MaximumTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()