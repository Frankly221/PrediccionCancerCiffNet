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

# CONFIGURACIÓN MÁXIMA RTX 3070 - 8GB FULL
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.6'
os.environ['OMP_NUM_THREADS'] = '20'

# Configuraciones RTX 3070 MÁXIMAS
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class RTX3070MaxVRAMTrainer:
    """Trainer RTX 3070 MÁXIMO - USAR 7-7.5GB VRAM"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN MÁXIMA 8GB
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # 95% = 7.6GB
            
            print(f"🔥 RTX 3070 MÁXIMO: Target 7-7.5GB VRAM (vs 2.9GB actual)")
        
        # Channels last ACTIVADO para máximo rendimiento
        self.model = self.model.to(memory_format=torch.channels_last)
        print("✅ Channels last: ACTIVADO para máximo throughput")
        
        # AMP MÁXIMO para aprovechar tensor cores
        self.scaler = GradScaler(
            init_scale=65536.0,   # Scale normal para estabilidad
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000
        )
        print("✅ AMP MÁXIMO: Tensor cores full speed")
        
        # Optimizer con todas las optimizaciones
        optimizer_kwargs = {
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'amsgrad': False
        }
        
        if hasattr(torch.optim.AdamW, 'foreach'):
            optimizer_kwargs['foreach'] = True
            print("✅ AdamW foreach: ACTIVADO")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-7
        )
        
        # Loss
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=self.class_weights)
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []
        self.vram_usage = []
        self.gpu_utilization = []
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        
        print(f"🚀 RTX 3070 MAX VRAM Trainer inicializado:")
        print(f"   ACTUAL: 2.9GB VRAM (36% usage) ❌")
        print(f"   TARGET: 7.0-7.5GB VRAM (87-94% usage) 🎯")
        print(f"   POTENCIAL: +142% más VRAM")
        print(f"   Batch size: {train_loader.batch_size}")
        
    def monitor_gpu_max(self):
        """Monitor GPU con foco en VRAM máximo"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_percent = (memory_allocated / 8.0) * 100
            utilization = min(99, memory_percent * 1.02)
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                util_gpu = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                pynvml.nvmlShutdown()
            except:
                temp = 0
                power_draw = 0
                util_gpu = 0
            
            return {
                'memory_allocated_gb': memory_allocated,
                'memory_reserved_gb': memory_reserved,
                'memory_percent': memory_percent,
                'estimated_utilization': utilization,
                'real_gpu_util': util_gpu,
                'temperature_c': temp,
                'power_draw_w': power_draw,
                'vram_efficiency': memory_percent / 100  # 0.0 to 1.0
            }
        return {
            'memory_allocated_gb': 0, 'memory_reserved_gb': 0,
            'memory_percent': 0, 'estimated_utilization': 0,
            'real_gpu_util': 0, 'temperature_c': 0, 'power_draw_w': 0,
            'vram_efficiency': 0
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
    
    def train_epoch_max_vram(self, epoch):
        """Entrenamiento MÁXIMO VRAM - 7GB+ target"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        gpu_stats_epoch = []
        vram_usage_epoch = []
        
        # Gradient accumulation MÁXIMO
        accumulation_steps = self.config.get('gradient_accumulation_steps', 2)  # Menos accumulation, más batch
        
        pbar = tqdm(self.train_loader, desc=f"🔥 MAX VRAM Epoch {epoch+1}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Transfer con channels_last MÁXIMO
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass con máximo aprovechamiento
            with autocast(dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
            
            # Backward
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
            gpu_stats = self.monitor_gpu_max()
            gpu_stats_epoch.append(gpu_stats['real_gpu_util'])
            vram_usage_epoch.append(gpu_stats['memory_allocated_gb'])
            
            current_vram = gpu_stats['memory_allocated_gb']
            vram_efficiency = gpu_stats['vram_efficiency']
            
            # Progress bar MÁXIMO
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['real_gpu_util']:.0f}%",
                'VRAM': f"{current_vram:.1f}GB",
                'Eff': f"{vram_efficiency*100:.0f}%",
                'Temp': f"{gpu_stats['temperature_c']:.0f}°C"
            })
            
            # Feedback VRAM MÁXIMO cada 15 batches
            if batch_idx % 15 == 0:
                vram_vs_actual = current_vram - 2.9
                vram_target_progress = (current_vram / 7.0) * 100  # Target 7GB
                
                if current_vram >= 7.0:
                    status = "🔥 VRAM MÁXIMO"
                elif current_vram >= 6.0:
                    status = "🎯 VRAM EXCELENTE"
                elif current_vram >= 5.0:
                    status = "✅ VRAM BUENO"
                elif current_vram >= 4.0:
                    status = "🟡 VRAM MEJORANDO"
                else:
                    status = "🔴 VRAM INSUFICIENTE"
                
                print(f"\n🚀 Batch {batch_idx}: {status}")
                print(f"   VRAM: {current_vram:.1f}GB / 8GB ({vram_efficiency*100:.1f}%)")
                print(f"   vs Actual: +{vram_vs_actual:.1f}GB ({vram_vs_actual/2.9*100:+.0f}%)")
                print(f"   Target Progress: {vram_target_progress:.0f}% (Target: 7GB)")
                print(f"   GPU Util: {gpu_stats['real_gpu_util']:.0f}%")
                print(f"   Power: {gpu_stats['power_draw_w']:.0f}W")
                
                # ESCALADO DINÁMICO si VRAM es bajo
                if current_vram < 5.0 and batch_idx > 50:
                    print(f"⚠️ VRAM BAJO detectado - considera aumentar batch size")
            
            # Limpieza mínima para mantener VRAM alto
            if batch_idx % 150 == 0:
                torch.cuda.empty_cache()
        
        # Final step
        if len(self.train_loader) % accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        self.scheduler.step()
        
        # Métricas finales
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        epoch_time = time.time() - epoch_start_time
        throughput = len(self.train_loader.dataset) / epoch_time
        avg_gpu_util = np.mean(gpu_stats_epoch) if gpu_stats_epoch else 0
        avg_vram = np.mean(vram_usage_epoch) if vram_usage_epoch else 0
        max_vram = max(vram_usage_epoch) if vram_usage_epoch else 0
        
        final_gpu_stats = self.monitor_gpu_max()
        
        return avg_loss, accuracy, throughput, final_gpu_stats, avg_gpu_util, avg_vram, max_vram
    
    def validate(self, epoch):
        """Validación eficiente"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
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
        print(f"🔥 INICIANDO RTX 3070 MÁXIMO VRAM TRAINING...")
        print(f"🎯 OBJETIVO: 7-7.5GB VRAM (vs 2.9GB actual)")
        print(f"📈 POTENCIAL: +142% más VRAM usage")
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train MAX VRAM
            results = self.train_epoch_max_vram(epoch)
            train_loss, train_acc, throughput, gpu_stats, avg_gpu_util, avg_vram, max_vram = results
            
            # Validate
            val_loss, val_acc, melanoma_recall, melanoma_precision = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            current_vram = gpu_stats['memory_allocated_gb']
            vram_vs_actual = current_vram - 2.9
            vram_efficiency = (current_vram / 8.0) * 100
            vram_improvement = (vram_vs_actual / 2.9) * 100
            
            print(f"\n{'='*130}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - RTX 3070 MÁXIMO VRAM")
            print(f"{'='*130}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⏱️  Tiempo: {epoch_time:.1f}s | Throughput: {throughput:.0f} samples/sec")
            print(f"")
            print(f"🚀 RTX 3070 MAXIMUM PERFORMANCE ANALYSIS:")
            print(f"   📊 VRAM CURRENT: {current_vram:.1f}GB / 8GB ({vram_efficiency:.1f}%)")
            print(f"   📈 VRAM MAX EPOCH: {max_vram:.1f}GB")
            print(f"   📉 VRAM AVG EPOCH: {avg_vram:.1f}GB")
            print(f"   🔥 vs ACTUAL (2.9GB): +{vram_vs_actual:.1f}GB ({vram_improvement:+.0f}%)")
            print(f"   🎯 Target Progress: {(current_vram/7.0)*100:.0f}% (Target: 7GB)")
            print(f"   ⚡ GPU Utilization: {avg_gpu_util:.0f}%")
            print(f"   🌡️  Temperature: {gpu_stats['temperature_c']:.0f}°C")
            print(f"   ⚡ Power: {gpu_stats['power_draw_w']:.0f}W")
            
            # VRAM Status Analysis
            if current_vram >= 7.0:
                print(f"   🔥 STATUS: VRAM MÁXIMO ALCANZADO!")
            elif current_vram >= 6.0:
                print(f"   🎯 STATUS: VRAM EXCELENTE - Cerca del máximo")
            elif current_vram >= 5.0:
                print(f"   ✅ STATUS: VRAM BUENO - Progreso sólido")
            elif current_vram >= 4.0:
                print(f"   🟡 STATUS: VRAM MEJORANDO - Continuar escalando")
            else:
                print(f"   🔴 STATUS: VRAM INSUFICIENTE - Revisar configuración")
            
            # Efficiency recommendations
            if vram_efficiency < 70:
                print(f"   💡 RECOMENDACIÓN: Incrementar batch size para más VRAM")
            elif vram_efficiency > 90:
                print(f"   ⚠️ ADVERTENCIA: VRAM muy alto - monitorear estabilidad")
            else:
                print(f"   ✅ OPTIMAL: VRAM en rango eficiente")
            
            # Save best model
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'vram_gb': current_vram,
                    'max_vram_gb': max_vram,
                    'gpu_util': avg_gpu_util,
                    'vram_efficiency': vram_efficiency,
                    'config': self.config
                }, 'best_max_vram_model.pth')
                print(f"   ✅ BEST MODEL: {val_acc:.2f}% | VRAM {current_vram:.1f}GB")

def main():
    """RTX 3070 MÁXIMO VRAM Training"""
    print("🔥 RTX 3070 MÁXIMO VRAM TRAINING")
    print("🎯 OBJETIVO: USAR 7-7.5GB de los 8GB disponibles")
    print("📊 ACTUAL: 2.9GB (36% usage) ❌")
    print("📈 TARGET: 7.0GB (87% usage) 🎯")
    print("💥 POTENCIAL: +142% más VRAM")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración MÁXIMA
    training_config = {
        'learning_rate': 4e-5,  # Ligeramente conservador para batch grande
        'weight_decay': 3e-4,
        'epochs': 40,
        'early_stopping_patience': 8,
        'gradient_clipping': 0.8,
        'gradient_accumulation_steps': 2  # Menos accumulation, más batch directo
    }
    
    # BATCH SIZE MÁXIMO - Empezar conservador e ir subiendo
    batch_size = 64  # 🔥 EMPEZAR AGRESIVO
    
    print(f"⚡ CONFIGURACIÓN MÁXIMA:")
    print(f"   Batch size: {batch_size} (vs 2.9GB actual)")
    print(f"   Gradient accumulation: 2")
    print(f"   Batch efectivo: 128")
    print(f"   Strategy: MÁXIMO VRAM possible")
    print(f"   Target: 7.0-7.5GB VRAM")
    
    # Cargar datos
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    try:
        train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
            csv_file, 
            image_folders, 
            batch_size=batch_size
        )
        
        print(f"✅ DataLoaders creados con batch {batch_size}")
        print(f"   Expected VRAM: ~5-7GB (target)")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"🔴 OOM con batch {batch_size}, probando batch {batch_size//2}")
            batch_size = batch_size // 2
            train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
                csv_file, image_folders, batch_size=batch_size
            )
        else:
            raise e
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    print(f"✅ Modelo creado para {num_classes} classes")
    
    # Entrenar MÁXIMO VRAM
    trainer = RTX3070MaxVRAMTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    print(f"\n🔥 INICIANDO MÁXIMO APROVECHAMIENTO RTX 3070...")
    print(f"📊 Monitor: watch -n 1 'nvidia-smi'")
    print(f"🎯 Buscar: 7000+ MB VRAM usage")
    
    trainer.train()

if __name__ == "__main__":
    main()