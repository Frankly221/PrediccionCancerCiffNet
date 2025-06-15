import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN RTX 3070 OPTIMIZADA
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
os.environ['OMP_NUM_THREADS'] = '20'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class RTX3070CleanTrainer:
    """Trainer RTX 3070 LIMPIO - 6GB VRAM optimizado"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN OPTIMIZADA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.96)
        
        # Optimizaciones
        self.model = self.model.to(memory_format=torch.channels_last)
        
        self.scaler = GradScaler(
            init_scale=32768.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1500
        )
        
        optimizer_kwargs = {
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }
        
        if hasattr(torch.optim.AdamW, 'foreach'):
            optimizer_kwargs['foreach'] = True
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optimizer_kwargs)
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-7
        )
        
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=self.class_weights)
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        
        print(f"✅ RTX 3070 Trainer inicializado - Batch size: {train_loader.batch_size}")
        
    def monitor_gpu(self):
        """Monitor GPU simple"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_percent = (memory_allocated / 8.0) * 100
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                util_gpu = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                pynvml.nvmlShutdown()
            except:
                temp = 0
                util_gpu = 0
            
            return {
                'memory_gb': memory_allocated,
                'memory_percent': memory_percent,
                'gpu_util': util_gpu,
                'temperature': temp
            }
        return {'memory_gb': 0, 'memory_percent': 0, 'gpu_util': 0, 'temperature': 0}
    
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
    
    def train_epoch(self, epoch):
        """Entrenamiento limpio"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        pbar = tqdm(self.train_loader, desc=f"Época {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Simple progress bar
            if batch_idx % 20 == 0:
                gpu_stats = self.monitor_gpu()
                pbar.set_postfix({
                    'Loss': f"{loss.item() * accumulation_steps:.4f}",
                    'Acc': f"{100.*correct/total:.1f}%",
                    'GPU': f"{gpu_stats['gpu_util']:.0f}%",
                    'VRAM': f"{gpu_stats['memory_gb']:.1f}GB"
                })
        
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, accuracy, epoch_time
    
    def validate(self, epoch):
        """Validación simple"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validando"):
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
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision
    
    def train(self):
        print("🚀 Iniciando entrenamiento RTX 3070...")
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_time = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, melanoma_recall, melanoma_precision = self.validate(epoch)
            
            # GPU stats
            gpu_stats = self.monitor_gpu()
            
            # Calcular throughput
            total_samples = len(self.train_loader.dataset)
            throughput = total_samples / train_time
            
            # Print clean stats
            print(f"\n{'='*80}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']}")
            print(f"{'='*80}")
            print(f"🔥 Entrenamiento:")
            print(f"   Loss: {train_loss:.4f}")
            print(f"   Accuracy: {train_acc:.2f}%")
            print(f"   Tiempo: {train_time:.1f}s")
            print(f"   Throughput: {throughput:.0f} samples/sec")
            print(f"")
            print(f"📊 Validación:")
            print(f"   Loss: {val_loss:.4f}")
            print(f"   Accuracy: {val_acc:.2f}%")
            print(f"")
            print(f"🩺 Melanoma:")
            print(f"   Recall: {melanoma_recall*100:.2f}%")
            print(f"   Precision: {melanoma_precision*100:.2f}%")
            print(f"")
            print(f"⚡ Sistema:")
            print(f"   GPU: {gpu_stats['gpu_util']:.0f}%")
            print(f"   VRAM: {gpu_stats['memory_gb']:.1f}GB ({gpu_stats['memory_percent']:.0f}%)")
            print(f"   Temperatura: {gpu_stats['temperature']:.0f}°C")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best models
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'config': self.config
                }, 'best_model_accuracy.pth')
                print(f"✅ Mejor modelo guardado: {val_acc:.2f}%")
            
            if melanoma_recall > self.best_melanoma_recall:
                self.best_melanoma_recall = melanoma_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'config': self.config
                }, 'best_model_melanoma.pth')
                print(f"✅ Mejor melanoma guardado: {melanoma_recall*100:.2f}%")

def main():
    """Entrenamiento RTX 3070 limpio"""
    print("🚀 RTX 3070 Training - Logs limpios")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración
    training_config = {
        'learning_rate': 3e-5,
        'weight_decay': 2e-4,
        'epochs': 40,
        'early_stopping_patience': 8,
        'gradient_clipping': 0.6,
        'gradient_accumulation_steps': 1
    }
    
    # Batch size optimizado para 6GB VRAM
    batch_size = 84
    
    print(f"Configuración:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Épocas: {training_config['epochs']}")
    
    # Cargar datos
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file, 
        image_folders, 
        batch_size=batch_size
    )
    
    print(f"✅ Datos cargados:")
    print(f"   Train: {len(train_loader.dataset)} samples")
    print(f"   Val: {len(val_loader.dataset)} samples")
    print(f"   Clases: {len(label_encoder.classes_)}")
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    # Entrenar
    trainer = RTX3070CleanTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()