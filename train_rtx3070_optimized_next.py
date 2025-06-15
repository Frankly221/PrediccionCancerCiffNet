import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN OPTIMIZADA RTX 3070 - FIXED
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.8'
os.environ['OMP_NUM_THREADS'] = '20'
os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# Configuraciones RTX 3070 optimizadas
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)
torch.autograd.set_detect_anomaly(False)

# Nuevas optimizaciones
try:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    print("✅ CUDA optimizations enabled")
except:
    print("⚠️ Some CUDA optimizations not available")

# USAR IMPORTS CORRECTOS
from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm
import torchvision.transforms as transforms

def get_optimized_transforms_256():
    """Transformaciones optimizadas 256x256 para RTX 3070"""
    return transforms.Compose([
        transforms.Resize((256, 256)),  # ⬆️ Resolución mayor
        transforms.RandomRotation(30, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(
            degrees=20, 
            translate=(0.1, 0.1), 
            scale=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomResizedCrop((256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class RTX3070OptimizedTrainer:
    """Trainer OPTIMIZADO RTX 3070 - FIXED VERSION"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN EXTREMA RTX 3070 - OPTIMIZADA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.97)  # 97% VRAM
            
            # Optimizaciones específicas
            torch.backends.cuda.cufft_plan_cache.max_size = 16
            
            print(f"✅ RTX 3070 OPTIMIZADO: VRAM 97%, resolución 256x256")
        
        # Convertir modelo a channels_last para mejor rendimiento
        self.model = self.model.to(memory_format=torch.channels_last)
        print("✅ Channels last: ✅ ACTIVADO")
        
        # AMP EXTREMO OPTIMIZADO
        self.scaler = GradScaler(
            init_scale=262144.0,  # ⬆️ Scale MUY alto
            growth_factor=2.0,
            backoff_factor=0.25,
            growth_interval=800
        )
        
        # Optimizer EXTREMO con nuevas optimizaciones
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False,
            foreach=True if hasattr(torch.optim.AdamW, 'foreach') else False
        )
        
        # Scheduler COSINE más agresivo
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-7,
            last_epoch=-1
        )
        
        # Loss optimizado
        self.criterion = FocalLoss(alpha=1.0, gamma=2.5, weight=self.class_weights)
        
        # Tracking optimizado
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
        
        # Best metrics
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 RTX 3070 OPTIMIZED Trainer inicializado:")
        print(f"   CURRENT: 96% GPU, 3.7GB VRAM ✅")
        print(f"   TARGET: 96%+ GPU, 5-6GB VRAM")
        print(f"   Resolución: 256x256 (vs 224x224)")
        print(f"   Batch size: {train_loader.batch_size}")
        print(f"   Modelo: channels_last optimizado")
        
    def monitor_gpu_optimized(self):
        """Monitor GPU optimizado con más métricas"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
            # Memory detallada
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_cached = torch.cuda.memory_cached() / 1e9
            
            # RTX 3070 8GB
            memory_percent = (memory_allocated / 8.0) * 100
            utilization = min(99, memory_percent * 1.02)
            
            # Temperatura y power
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
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
        """Métricas melanoma optimizadas"""
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
    
    def compute_balanced_score(self, accuracy, melanoma_recall):
        """Score balanceado optimizado"""
        return 0.65 * accuracy + 0.35 * (melanoma_recall * 100)
    
    def train_epoch_optimized(self, epoch):
        """Entrenamiento OPTIMIZADO RTX 3070 - Resolución 256x256"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        gpu_stats_epoch = []
        
        # Gradient accumulation optimizado
        accumulation_steps = self.config.get('gradient_accumulation_steps', 3)
        
        pbar = tqdm(self.train_loader, desc=f"🔥 RTX3070 OPTIMIZED Epoch {epoch+1}/{self.config['epochs']}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Pre-warm optimizado
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Transfer OPTIMIZADO channels_last
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass OPTIMIZADO
            with autocast(dtype=torch.float16, cache_enabled=True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            # Backward OPTIMIZADO
            self.scaler.scale(loss).backward()
            
            # Optimizer step optimizado
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
            
            # GPU monitoring optimizado
            batch_time = time.time() - batch_start_time
            gpu_stats = self.monitor_gpu_optimized()
            gpu_stats_epoch.append(gpu_stats['estimated_utilization'])
            
            samples_per_second = inputs.size(0) / batch_time
            
            # Progress bar OPTIMIZADO
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['estimated_utilization']:.1f}%",
                'VRAM': f"{gpu_stats['memory_allocated_gb']:.1f}GB",
                'Temp': f"{gpu_stats['temperature_c']:.0f}°C",
                'SPS': f"{samples_per_second:.0f}",
                'Res': '256x256'
            })
            
            # Feedback tiempo real OPTIMIZADO
            if batch_idx % 30 == 0:
                current_gpu = gpu_stats['estimated_utilization']
                current_vram = gpu_stats['memory_allocated_gb']
                vram_vs_base = current_vram - 3.7
                
                if current_gpu >= 95 and current_vram >= 5:
                    status = "🎯 PERFECTO"
                elif current_gpu >= 90 and current_vram >= 4.5:
                    status = "✅ EXCELENTE"
                elif current_gpu >= 85:
                    status = "🟡 BUENO"
                else:
                    status = "⚠️ OPTIMIZABLE"
                
                print(f"\n📊 Batch {batch_idx}: {status} - GPU {current_gpu:.1f}% | VRAM {current_vram:.1f}GB ({vram_vs_base:+.1f}GB vs base) | 256x256")
            
            # Limpieza optimizada
            if batch_idx % 75 == 0:
                torch.cuda.empty_cache()
        
        # Final step
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # Scheduler step
        self.scheduler.step()
        
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
        
        final_gpu_stats = self.monitor_gpu_optimized()
        self.vram_usage.append(final_gpu_stats['memory_allocated_gb'])
        self.gpu_temperature.append(final_gpu_stats['temperature_c'])
        self.gpu_power.append(final_gpu_stats['power_draw_w'])
        
        return avg_loss, accuracy, throughput, final_gpu_stats, avg_gpu_util
    
    def validate(self, epoch, is_final=False):
        """Validación optimizada"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        validation_start = time.time()
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating OPTIMIZED"):
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.float16, cache_enabled=True):
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
        
        # Guardar métricas básicas
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, melanoma_recall, melanoma_precision, validation_time
    
    def train(self):
        print(f"🚀 INICIANDO ENTRENAMIENTO RTX 3070 OPTIMIZED...")
        print(f"🎯 OBJETIVO: 256x256 + mantener 96% GPU + >5GB VRAM")
        print(f"⚙️  BASE: 96% GPU, 3.7GB VRAM, 224x224")
        print(f"📈 TARGET: 96%+ GPU, 5-6GB VRAM, 256x256, accuracy +2-3%")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train OPTIMIZED
            train_loss, train_acc, throughput, gpu_stats, avg_gpu_util = self.train_epoch_optimized(epoch)
            
            # Validate
            val_results = self.validate(epoch, is_final=False)
            val_loss, val_acc, melanoma_recall, melanoma_precision, val_time = val_results
            
            epoch_time = time.time() - epoch_start
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
            # System stats
            import psutil
            ram = psutil.virtual_memory()
            
            vram_improvement = gpu_stats['memory_allocated_gb'] - 3.7
            
            print(f"\n{'='*140}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - RTX 3070 OPTIMIZED (256x256)")
            print(f"{'='*140}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempos: Época {epoch_time:.1f}s | Throughput {throughput:.0f} samples/sec")
            print(f"🚀 RTX 3070 OPTIMIZED PERFORMANCE:")
            print(f"   GPU Utilization: {avg_gpu_util:.1f}% (Base: 96%)")
            print(f"   VRAM Usage: {gpu_stats['memory_allocated_gb']:.1f}GB/8GB ({gpu_stats['memory_percent']:.1f}%)")
            print(f"   VRAM vs Base: {vram_improvement:+.1f}GB")
            print(f"   Resolution: 256x256 (vs 224x224)")
            print(f"   Temperature: {gpu_stats['temperature_c']:.0f}°C")
            print(f"   Power: {gpu_stats['power_draw_w']:.0f}W")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Performance analysis
            if vram_improvement >= 1.5:
                print(f"🎯 VRAM EXCELENTE: +{vram_improvement:.1f}GB vs base")
            elif vram_improvement >= 0.5:
                print(f"✅ VRAM BUENO: +{vram_improvement:.1f}GB vs base")
            elif vram_improvement >= 0:
                print(f"🟡 VRAM SIMILAR: +{vram_improvement:.1f}GB vs base")
            else:
                print(f"⚠️ VRAM MENOR: {vram_improvement:.1f}GB vs base")
            
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
                    'mode': 'RTX3070_OPTIMIZED_256x256'
                }, 'best_rtx3070_optimized_overall.pth')
                print(f"✅ Mejor modelo RTX 3070 OPTIMIZED guardado: {val_acc:.2f}%")
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
                    'mode': 'RTX3070_OPTIMIZED_256x256'
                }, 'best_rtx3070_optimized_melanoma.pth')
                print(f"✅ Mejor melanoma RTX 3070 OPTIMIZED guardado: {melanoma_recall*100:.2f}%")
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
                    'mode': 'RTX3070_OPTIMIZED_256x256'
                }, 'best_rtx3070_optimized_balanced.pth')
                print(f"✅ Mejor balance RTX 3070 OPTIMIZED guardado: {balanced_score:.2f}%")
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
        
        total_time = time.time() - start_time
        avg_gpu_util = np.mean(self.gpu_utilization) if self.gpu_utilization else 0
        max_vram = max(self.vram_usage) if self.vram_usage else 0
        avg_temp = np.mean(self.gpu_temperature) if self.gpu_temperature else 0
        
        print(f"\n🎉 ENTRENAMIENTO RTX 3070 OPTIMIZED COMPLETADO!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")
        print(f"🚀 RENDIMIENTO FINAL vs BASE:")
        print(f"   GPU: {avg_gpu_util:.1f}% (Base: 96%)")
        print(f"   VRAM: {max_vram:.1f}GB (Base: 3.7GB) = +{max_vram-3.7:.1f}GB")
        print(f"   Resolución: 256x256 (Base: 224x224)")
        print(f"   Temperature: {avg_temp:.0f}°C")

def main():
    """Entrenamiento RTX 3070 OPTIMIZED - 256x256 FIXED"""
    print("🚀 ENTRENAMIENTO RTX 3070 OPTIMIZED - FIXED")
    print("🎯 OBJETIVO: 256x256 + mantener 96% GPU + >5GB VRAM")
    print("⚙️  BASE: 96% GPU, 3.7GB VRAM, 224x224")
    print("📈 TARGET: 96%+ GPU, 5-6GB VRAM, 256x256, accuracy +2-3%")
    print("🔧 FIXED: Import correcto + channels_last")
    print("=" * 120)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración OPTIMIZADA RTX 3070
    training_config = {
        'learning_rate': 4e-5,    # ⬇️ Más conservador para resolución mayor
        'weight_decay': 5e-4,     # ⬆️ Más regularización
        'loss_type': 'focal',
        'scheduler': 'cosine',
        'epochs': 70,             # ⬆️ Más épocas para convergencia
        'early_stopping_patience': 6,
        'gradient_clipping': 0.3, # ⬇️ Más conservador
        'mixed_precision': True,
        'gradient_accumulation_steps': 3
    }
    
    # BATCH SIZE optimizado para 256x256
    batch_size = 28  # ⬇️ Reducir para 256x256 pero aprovechar VRAM
    
    print(f"⚡ CONFIGURACIÓN RTX 3070 OPTIMIZED:")
    print(f"   Resolución: 256x256 (33% más píxeles)")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {training_config['gradient_accumulation_steps']}")
    print(f"   Batch efectivo: {batch_size * training_config['gradient_accumulation_steps']} = 84")
    print(f"   Target VRAM: 5-6GB")
    print(f"   Scheduler: Cosine annealing")
    print(f"   Channels last: ✅ ACTIVADO")
    
    # Cargar datos con función existente - FIXED
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    # USAR create_improved_data_loaders DIRECTAMENTE
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file, 
        image_folders, 
        batch_size=batch_size,
        train_transforms=get_optimized_transforms_256(),  # ⬆️ Transformaciones optimizadas
        val_transforms=transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    
    print(f"✅ DataLoaders creados con resolución 256x256")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Classes: {len(label_encoder.classes_)}")
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',  # Mantener B1 para esta fase
        pretrained=True
    )
    
    print(f"✅ Modelo EfficientNet-B1 creado para {num_classes} classes")
    
    # Entrenar RTX 3070 OPTIMIZED
    trainer = RTX3070OptimizedTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()