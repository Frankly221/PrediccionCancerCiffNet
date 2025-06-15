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

# CONFIGURACIÓN NIVEL 2 RTX 3070 - 6-7GB TARGET
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16,garbage_collection_threshold:0.5'
os.environ['OMP_NUM_THREADS'] = '20'

# Configuraciones nivel 2
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(20)

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss
from tqdm import tqdm

class RTX3070Level2Trainer:
    """Trainer RTX 3070 NIVEL 2 - 4.8GB → 6.5GB VRAM"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # CONFIGURACIÓN NIVEL 2 - 96% VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.96)  # 96% = 7.7GB
            
            print(f"🔥 RTX 3070 NIVEL 2: Target 6.5-7GB VRAM")
            print(f"📈 PROGRESO: 4.8GB → 6.5GB (+35% más)")
        
        # Channels last MÁXIMO
        self.model = self.model.to(memory_format=torch.channels_last)
        print("✅ Channels last: MÁXIMO rendimiento")
        
        # AMP optimizado para nivel 2
        self.scaler = GradScaler(
            init_scale=32768.0,   # Scale medio para estabilidad
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=1500
        )
        
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
            optimizer_kwargs['capturable'] = True if hasattr(torch.optim.AdamW, 'capturable') else False
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs
        )
        
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
        self.vram_usage = []
        self.gpu_utilization = []
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        
        print(f"🚀 RTX 3070 NIVEL 2 Trainer inicializado:")
        print(f"   PROGRESO: 2.9GB → 4.8GB ✅ (+65%)")
        print(f"   SIGUIENTE: 4.8GB → 6.5GB 🎯 (+35%)")
        print(f"   Batch size: {train_loader.batch_size}")
        
    def monitor_gpu_level2(self):
        """Monitor GPU nivel 2"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            memory_percent = (memory_allocated / 8.0) * 100
            
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
                'real_gpu_util': util_gpu,
                'temperature_c': temp,
                'power_draw_w': power_draw,
                'progress_to_target': (memory_allocated / 6.5) * 100  # Target 6.5GB
            }
        return {
            'memory_allocated_gb': 0, 'memory_reserved_gb': 0,
            'memory_percent': 0, 'real_gpu_util': 0,
            'temperature_c': 0, 'power_draw_w': 0,
            'progress_to_target': 0
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
    
    def train_epoch_level2(self, epoch):
        """Entrenamiento NIVEL 2 - 6.5GB target"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        gpu_stats_epoch = []
        vram_usage_epoch = []
        
        # Gradient accumulation MÍNIMO para máximo batch directo
        accumulation_steps = self.config.get('gradient_accumulation_steps', 1)  # Mínimo accumulation
        
        pbar = tqdm(self.train_loader, desc=f"🔥 NIVEL 2 Epoch {epoch+1}")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            batch_start_time = time.time()
            
            # Transfer NIVEL 2
            inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
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
            
            # GPU monitoring NIVEL 2
            gpu_stats = self.monitor_gpu_level2()
            gpu_stats_epoch.append(gpu_stats['real_gpu_util'])
            vram_usage_epoch.append(gpu_stats['memory_allocated_gb'])
            
            current_vram = gpu_stats['memory_allocated_gb']
            progress_target = gpu_stats['progress_to_target']
            
            # Progress bar NIVEL 2
            pbar.set_postfix({
                'Loss': f"{loss.item() * accumulation_steps:.4f}",
                'Acc': f"{100.*correct/total:.1f}%",
                'GPU': f"{gpu_stats['real_gpu_util']:.0f}%",
                'VRAM': f"{current_vram:.1f}GB",
                'Prog': f"{progress_target:.0f}%",
                'Temp': f"{gpu_stats['temperature_c']:.0f}°C"
            })
            
            # Feedback NIVEL 2 cada 12 batches
            if batch_idx % 12 == 0:
                vram_vs_level1 = current_vram - 4.8
                vram_remaining = 6.5 - current_vram
                
                if current_vram >= 6.5:
                    status = "🔥 NIVEL 2 COMPLETO"
                elif current_vram >= 6.0:
                    status = "🎯 NIVEL 2 CASI"
                elif current_vram >= 5.5:
                    status = "✅ NIVEL 2 PROGRESO"
                elif current_vram >= 5.0:
                    status = "🟡 NIVEL 2 INICIO"
                else:
                    status = "⚠️ NIVEL 1 AÚN"
                
                print(f"\n🚀 Batch {batch_idx}: {status}")
                print(f"   📊 VRAM: {current_vram:.1f}GB / 6.5GB target ({progress_target:.0f}%)")
                print(f"   📈 vs Nivel 1: +{vram_vs_level1:.1f}GB")
                print(f"   🎯 Restante: {vram_remaining:.1f}GB para objetivo")
                print(f"   ⚡ GPU: {gpu_stats['real_gpu_util']:.0f}% | Power: {gpu_stats['power_draw_w']:.0f}W")
                
                # Recomendación dinámica
                if current_vram < 5.5 and batch_idx > 30:
                    print(f"   💡 SUGERENCIA: Batch size puede subir más (target: 6.5GB)")
                elif current_vram >= 6.5:
                    print(f"   🎉 OBJETIVO ALCANZADO! Nivel 2 completo")
            
            # Limpieza mínima
            if batch_idx % 200 == 0:
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
        
        final_gpu_stats = self.monitor_gpu_level2()
        
        return avg_loss, accuracy, throughput, final_gpu_stats, avg_gpu_util, avg_vram, max_vram
    
    def validate(self, epoch):
        """Validación nivel 2"""
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
        print(f"🔥 INICIANDO RTX 3070 NIVEL 2...")
        print(f"📊 PROGRESO: 2.9GB → 4.8GB ✅")
        print(f"🎯 OBJETIVO: 4.8GB → 6.5GB")
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train NIVEL 2
            results = self.train_epoch_level2(epoch)
            train_loss, train_acc, throughput, gpu_stats, avg_gpu_util, avg_vram, max_vram = results
            
            # Validate
            val_loss, val_acc, melanoma_recall, melanoma_precision = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            current_vram = gpu_stats['memory_allocated_gb']
            level2_progress = (current_vram - 4.8) / (6.5 - 4.8) * 100  # Progreso Nivel 2
            total_progress = (current_vram / 6.5) * 100  # Progreso total
            
            print(f"\n{'='*130}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - RTX 3070 NIVEL 2")
            print(f"{'='*130}")
            print(f"🔥 Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
            print(f"📊 Val: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
            print(f"🩺 Melanoma: Recall {melanoma_recall*100:.2f}% | Precision {melanoma_precision*100:.2f}%")
            print(f"⏱️  Tiempo: {epoch_time:.1f}s | Throughput: {throughput:.0f} samples/sec")
            print(f"")
            print(f"🚀 RTX 3070 NIVEL 2 ANÁLISIS:")
            print(f"   📊 VRAM ACTUAL: {current_vram:.1f}GB / 6.5GB target")
            print(f"   📈 VRAM MAX ÉPOCA: {max_vram:.1f}GB")
            print(f"   📉 VRAM AVG ÉPOCA: {avg_vram:.1f}GB")
            print(f"   🎯 Progreso Nivel 2: {level2_progress:.0f}% (4.8GB → 6.5GB)")
            print(f"   📊 Progreso Total: {total_progress:.0f}% (0GB → 6.5GB)")
            print(f"   ⚡ GPU Utilization: {avg_gpu_util:.0f}%")
            print(f"   🌡️  Temperature: {gpu_stats['temperature_c']:.0f}°C")
            print(f"   ⚡ Power: {gpu_stats['power_draw_w']:.0f}W")
            
            # Status Level 2
            if current_vram >= 6.5:
                print(f"   🔥 STATUS: NIVEL 2 COMPLETADO! Target alcanzado")
                print(f"   🚀 SIGUIENTE: Considerar Nivel 3 (7-7.5GB) o modelo mayor")
            elif current_vram >= 6.0:
                print(f"   🎯 STATUS: NIVEL 2 CASI COMPLETO - Muy cerca!")
            elif current_vram >= 5.5:
                print(f"   ✅ STATUS: NIVEL 2 BUEN PROGRESO - Continuar")
            elif current_vram >= 5.0:
                print(f"   🟡 STATUS: NIVEL 2 INICIO - Escalando")
            else:
                print(f"   ⚠️ STATUS: AÚN EN NIVEL 1 - Incrementar batch")
            
            # Recomendaciones dinámicas
            if current_vram < 5.5:
                recommended_batch = int(self.train_loader.batch_size * 1.3)
                print(f"   💡 RECOMENDACIÓN: Incrementar batch a ~{recommended_batch}")
            elif current_vram >= 6.5:
                print(f"   🎉 FELICITACIONES: RTX 3070 aprovechada al máximo!")
            
            # Save best model
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_acc': val_acc,
                    'vram_gb': current_vram,
                    'max_vram_gb': max_vram,
                    'level2_progress': level2_progress,
                    'gpu_util': avg_gpu_util,
                    'config': self.config
                }, 'best_level2_model.pth')
                print(f"   ✅ BEST NIVEL 2: {val_acc:.2f}% | VRAM {current_vram:.1f}GB")

def main():
    """RTX 3070 NIVEL 2 Training - 4.8GB → 6.5GB"""
    print("🔥 RTX 3070 NIVEL 2 TRAINING")
    print("📊 BASE: 2.9GB → 4.8GB ✅ COMPLETADO")
    print("🎯 NIVEL 2: 4.8GB → 6.5GB (81% usage)")
    print("📈 INCREMENTO: +35% más VRAM")
    print("=" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuración NIVEL 2
    training_config = {
        'learning_rate': 3e-5,  # Más conservador para batch mayor
        'weight_decay': 2e-4,
        'epochs': 40,
        'early_stopping_patience': 8,
        'gradient_clipping': 0.6,
        'gradient_accumulation_steps': 1  # Mínimo accumulation, máximo batch directo
    }
    
    # BATCH SIZE NIVEL 2 - Incremento desde actual
    current_batch = 64  # Asumiendo que usas ~64 para 4.8GB
    level2_batch = 84   # +31% incremento para llegar a 6.5GB
    
    print(f"⚡ CONFIGURACIÓN NIVEL 2:")
    print(f"   Batch size: {level2_batch} (vs {current_batch} actual)")
    print(f"   Incremento: +{level2_batch-current_batch} batches (+{((level2_batch/current_batch)-1)*100:.0f}%)")
    print(f"   Gradient accumulation: 1 (máximo batch directo)")
    print(f"   Target VRAM: 6.5GB")
    print(f"   Expected: 6.0-6.5GB VRAM")
    
    # Cargar datos
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    try:
        train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
            csv_file, 
            image_folders, 
            batch_size=level2_batch
        )
        
        print(f"✅ DataLoaders NIVEL 2 creados con batch {level2_batch}")
        print(f"   Expected VRAM: 6.0-6.5GB")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"🔴 OOM con batch {level2_batch}, probando batch {level2_batch-12}")
            level2_batch = level2_batch - 12
            train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
                csv_file, image_folders, batch_size=level2_batch
            )
            print(f"✅ Batch ajustado a {level2_batch}")
        else:
            raise e
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    print(f"✅ Modelo EfficientNet-B1 creado")
    
    # Entrenar NIVEL 2
    trainer = RTX3070Level2Trainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    print(f"\n🔥 INICIANDO NIVEL 2...")
    print(f"📊 Monitor: watch -n 1 'nvidia-smi'")
    print(f"🎯 Buscar: 6000-6500 MB VRAM")
    
    trainer.train()

if __name__ == "__main__":
    main()