import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm
import os
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN EXTREMA PARA 32GB RAM + RTX 3070
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '16'  # ⬆️ NUEVO - aprovechar más CPU threads

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configurar para MÁXIMA RAM usage
torch.set_num_threads(16)  # ⬆️ NUEVO - más threads CPU

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss

class ExtremeRAMGPUTrainer:
    """Trainer optimizado para 32GB RAM + RTX 3070"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # Configurar GPU + RAM para máximo uso
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.99)  # ⬆️ 99% VRAM
        
        # AMP con más precisión
        self.scaler = GradScaler()
        
        # Optimizer mejorado para 32GB RAM
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler más agresivo con más RAM
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=3,     # ⬇️ Más agresivo con mucha RAM
            min_lr=1e-8     # ⬇️ Puede ir más bajo
        )
        self.scheduler_verbose = True
        
        # Loss optimizado
        self.criterion = FocalLoss(
            alpha=1.0,
            gamma=2.0,
            weight=self.class_weights
        )
        
        # Tracking con más métricas (gracias a 32GB RAM)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.learning_rates = []
        self.gpu_memory_usage = []     # ⬆️ NUEVO
        self.ram_usage = []            # ⬆️ NUEVO
        self.batch_times = []          # ⬆️ NUEVO
        
        # Best metrics tracking
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 EXTREME RAM+GPU Trainer inicializado:")
        print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"   VRAM asignada: 99%")
        print(f"   RAM disponible: 32GB")
        print(f"   CPU threads: 16")
        print(f"   TF32: Habilitado")
        print(f"   Batch size: {train_loader.batch_size}")
        
    def get_ram_usage(self):
        """Monitor RAM usage"""
        import psutil
        return psutil.virtual_memory().percent
    
    def compute_melanoma_metrics(self, all_predicted, all_targets):
        """Calcular métricas específicas para melanoma"""
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
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_batch_times = []
        
        pbar = tqdm(self.train_loader, desc=f"🔥 32GB+RTX3070 Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start = time.time()
            
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Performance metrics
            batch_time = time.time() - batch_start
            epoch_batch_times.append(batch_time)
            
            gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            gpu_percent = (gpu_memory / 8.0) * 100  # RTX 3070 = 8GB
            ram_percent = self.get_ram_usage()
            
            # Update progress con TODAS las métricas
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_percent:.0f}%",
                'VRAM': f"{gpu_memory:.1f}GB",
                'RAM': f"{ram_percent:.0f}%",
                'Time': f"{batch_time:.2f}s",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Limpieza cada 10 batches (más agresivo con 32GB RAM)
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        avg_batch_time = np.mean(epoch_batch_times)
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        self.batch_times.append(avg_batch_time)
        self.gpu_memory_usage.append(gpu_memory)
        self.ram_usage.append(self.get_ram_usage())
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating 32GB+RTX3070"):
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
        
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, all_predicted, all_targets, melanoma_recall, melanoma_precision
    
    def compute_balanced_score(self, accuracy, melanoma_recall):
        """Score balanceado que prioriza melanoma"""
        return 0.6 * accuracy + 0.4 * (melanoma_recall * 100)
    
    def save_extreme_plots(self):
        """Guardar gráficos con métricas extras de 32GB RAM"""
        plt.figure(figsize=(24, 16))  # ⬆️ Más grande con 32GB RAM
        
        # Loss plot
        plt.subplot(3, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        plt.title('Loss - 32GB RAM + RTX 3070', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(3, 4, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='blue', linewidth=2)
        plt.plot(self.val_accuracies, label='Val Acc', color='red', linewidth=2)
        plt.title('Accuracy - 32GB RAM + RTX 3070', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma recall
        plt.subplot(3, 4, 3)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=2)
        plt.title('Melanoma Recall - 32GB Setup', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate
        plt.subplot(3, 4, 4)
        plt.plot(self.learning_rates, color='green', linewidth=2)
        plt.title('Learning Rate - 32GB Setup', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # ⬆️ NUEVOS PLOTS con 32GB RAM
        
        # GPU Memory Usage
        plt.subplot(3, 4, 5)
        plt.plot(self.gpu_memory_usage, color='orange', linewidth=2)
        plt.title('GPU Memory Usage', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('VRAM (GB)')
        plt.grid(True, alpha=0.3)
        
        # RAM Usage
        plt.subplot(3, 4, 6)
        plt.plot(self.ram_usage, color='purple', linewidth=2)
        plt.title('RAM Usage (32GB Total)', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('RAM (%)')
        plt.grid(True, alpha=0.3)
        
        # Batch Processing Time
        plt.subplot(3, 4, 7)
        plt.plot(self.batch_times, color='brown', linewidth=2)
        plt.title('Batch Processing Time', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Time (s)')
        plt.grid(True, alpha=0.3)
        
        # Performance Summary
        plt.subplot(3, 4, 8)
        epochs = range(1, len(self.val_accuracies) + 1)
        plt.plot(epochs, self.val_accuracies, label='Val Acc', linewidth=2)
        plt.plot(epochs, self.melanoma_recalls, label='Melanoma Recall', linewidth=2)
        balanced_scores = [self.compute_balanced_score(acc, mel_rec/100) for acc, mel_rec in zip(self.val_accuracies, self.melanoma_recalls)]
        plt.plot(epochs, balanced_scores, label='Balanced Score', linewidth=2, linestyle='--')
        plt.title('Performance Summary - 32GB', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # System Resource Usage
        plt.subplot(3, 4, 9)
        epochs = range(1, len(self.gpu_memory_usage) + 1)
        gpu_percent = [(mem/8.0)*100 for mem in self.gpu_memory_usage]
        plt.plot(epochs, gpu_percent, label='GPU Usage %', linewidth=2, color='red')
        plt.plot(epochs, self.ram_usage, label='RAM Usage %', linewidth=2, color='blue')
        plt.title('Resource Utilization', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Usage (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance per Resource
        plt.subplot(3, 4, 10)
        if len(self.val_accuracies) > 0 and len(self.gpu_memory_usage) > 0:
            efficiency = [acc / (gpu_mem + 0.1) for acc, gpu_mem in zip(self.val_accuracies, self.gpu_memory_usage)]
            plt.plot(efficiency, color='green', linewidth=2)
            plt.title('Efficiency (Acc/VRAM)', fontsize=14, fontweight='bold')
            plt.xlabel('Época')
            plt.ylabel('Efficiency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_32gb_rtx3070.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar métricas en CSV para análisis posterior
        metrics_df = pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies,
            'melanoma_recall': self.melanoma_recalls,
            'learning_rate': self.learning_rates,
            'gpu_memory_gb': self.gpu_memory_usage,
            'ram_usage_percent': self.ram_usage,
            'batch_time_sec': self.batch_times
        })
        metrics_df.to_csv('training_metrics_32gb_rtx3070.csv', index=False)
        print(f"📊 Métricas guardadas en training_metrics_32gb_rtx3070.csv")
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO EXTREMO - 32GB RAM + RTX 3070...")
        print(f"📊 Configuración: {self.config}")
        
        start_time = time.time()
        patience_counter = 0
        previous_lr = self.optimizer.param_groups[0]['lr']
        
        for epoch in range(self.config['epochs']):
            torch.cuda.empty_cache()
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, all_predicted, all_targets, melanoma_recall, melanoma_precision = self.validate()
            
            # Scheduler step (más agresivo con 32GB RAM)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_acc)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if self.scheduler_verbose and new_lr < previous_lr:
                print(f"📉 ReduceLROnPlateau: reducing learning rate to {new_lr:.2e}")
            previous_lr = new_lr
            
            epoch_time = time.time() - epoch_start
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
            # System stats
            gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            gpu_percent = (gpu_memory / 8.0) * 100
            ram_percent = self.get_ram_usage()
            
            # Print EXTREME stats
            print(f"\n{'='*90}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - 32GB RAM + RTX 3070 EXTREMO")
            print(f"{'='*90}")
            print(f"🔥 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"🩺 Melanoma Recall: {melanoma_recall*100:.2f}% | Precision: {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempo época: {epoch_time:.1f}s | Batch promedio: {np.mean(self.batch_times[-10:]) if self.batch_times else 0:.2f}s")
            print(f"🔥 GPU Usage: {gpu_percent:.0f}% | VRAM: {gpu_memory:.1f}GB/8GB")
            print(f"💾 RAM Usage: {ram_percent:.0f}% | Disponible: {32*(100-ram_percent)/100:.1f}GB")
            print(f"📈 LR actual: {self.optimizer.param_groups[0]['lr']:.2e}")
            
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
                    'config': self.config,
                    'system_specs': '32GB_RAM_RTX3070'
                }, 'best_32gb_rtx3070_overall.pth')
                print(f"✅ Mejor accuracy 32GB+RTX3070 guardado: {val_acc:.2f}%")
                saved_model = True
            
            if melanoma_recall > self.best_melanoma_recall:
                self.best_melanoma_recall = melanoma_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_melanoma_recall': melanoma_recall,
                    'val_acc': val_acc,
                    'config': self.config,
                    'system_specs': '32GB_RAM_RTX3070'
                }, 'best_32gb_rtx3070_melanoma.pth')
                print(f"✅ Mejor melanoma 32GB+RTX3070 guardado: {melanoma_recall*100:.2f}%")
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
                    'config': self.config,
                    'system_specs': '32GB_RAM_RTX3070'
                }, 'best_32gb_rtx3070_balanced.pth')
                print(f"✅ Mejor score balanceado 32GB+RTX3070 guardado: {balanced_score:.2f}%")
                saved_model = True
            
            if saved_model:
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"⏳ Paciencia: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping (más agresivo con recursos abundantes)
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                break
            
            # Guardar plots cada 3 épocas (más frecuente con 32GB RAM)
            if (epoch + 1) % 3 == 0:
                self.save_extreme_plots()
        
        total_time = time.time() - start_time
        self.save_extreme_plots()
        
        print(f"\n🎉 Entrenamiento EXTREMO 32GB+RTX3070 completado!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy general: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"💾 Máximo RAM usado: {max(self.ram_usage):.1f}%")
        print(f"🔥 Máximo GPU usado: {max([mem/8*100 for mem in self.gpu_memory_usage]):.1f}%")

def main():
    print("🚀 ENTRENAMIENTO EXTREMO - 32GB RAM + RTX 3070")
    print("=" * 70)
    
    # Verificar recursos
    import psutil
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"📊 VRAM asignada: 99%")
    
    # RAM info
    ram_info = psutil.virtual_memory()
    print(f"💾 RAM Total: {ram_info.total / 1e9:.1f} GB")
    print(f"💾 RAM Disponible: {ram_info.available / 1e9:.1f} GB")
    print(f"💾 RAM Uso actual: {ram_info.percent:.1f}%")
    
    # CPU info
    print(f"🔧 CPU Cores: {psutil.cpu_count()}")
    print(f"🔧 CPU Threads configurados: 16")
    print(f"⚡ TF32: Habilitado")
    
    # Configuración EXTREMA para 32GB RAM
    training_config = {
        'learning_rate': 5e-5,
        'weight_decay': 5e-4,
        'loss_type': 'focal',
        'scheduler': 'plateau',
        'epochs': 60,
        'early_stopping_patience': 6,  # ⬇️ Más agresivo con recursos abundantes
        'gradient_clipping': 0.3,
        'mixed_precision': True
    }
    
    # BATCH SIZE EXTREMO para 32GB RAM + RTX 3070
    batch_size = 28  # ⬆️ AUMENTADO de 24 a 28
    
    print(f"🎯 Configuración EXTREMA 32GB:")
    print(f"   Batch size: {batch_size} (MÁXIMO)")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print(f"   Weight decay: {training_config['weight_decay']}")
    print(f"   Early stopping: {training_config['early_stopping_patience']} épocas")
    print(f"   CPU threads: 16")
    print(f"   Workers: 20 (máximo)")
    
    # Cargar datos con configuración EXTREMA
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    print(f"📂 Cargando dataset EXTREMO 32GB...")
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file, image_folders, batch_size=batch_size
    )
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net EXTREMO para {num_classes} clases...")
    
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    # Entrenar con EXTREME trainer
    trainer = ExtremeRAMGPUTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()