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

# CONFIGURACIÓN RTX 3070 + ANÁLISIS MÉDICO
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
from medical_metrics_complete import MedicalMetricsComplete
from tqdm import tqdm

class RTX3070MedicalTrainer:
    """Trainer RTX 3070 con análisis médico completo para artículo"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # Configuración optimizada
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.96)
        
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
        
        # ANÁLISIS MÉDICO
        self.medical_evaluator = MedicalMetricsComplete(self.label_encoder.classes_)
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.medical_metrics_history = []
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"✅ RTX 3070 Medical Trainer inicializado")
        print(f"📊 Análisis médico completo activado")
        print(f"🎯 Clases: {len(self.label_encoder.classes_)}")
        
    def monitor_gpu(self):
        """Monitor GPU simple"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            
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
                'gpu_util': util_gpu,
                'temperature': temp
            }
        return {'memory_gb': 0, 'gpu_util': 0, 'temperature': 0}
    
    def train_epoch(self, epoch):
        """Entrenamiento con tracking médico"""
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
    
    def validate_with_medical_analysis(self, epoch, final_analysis=False):
        """Validación con análisis médico completo"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validación médica"):
                inputs = inputs.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(self.device, non_blocking=True)
                
                with autocast(dtype=torch.float16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Guardar probabilidades para AUC
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
        
        # ANÁLISIS MÉDICO COMPLETO
        y_true = np.array(all_targets)
        y_pred = np.array(all_predicted)
        y_pred_proba = np.array(all_probabilities)
        
        medical_metrics = self.medical_evaluator.compute_comprehensive_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        # Guardar historial
        self.medical_metrics_history.append({
            'epoch': epoch,
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'medical_metrics': medical_metrics
        })
        
        # Análisis detallado cada 5 épocas o final
        if epoch % 5 == 0 or final_analysis:
            print(f"\n🏥 ANÁLISIS MÉDICO DETALLADO - ÉPOCA {epoch+1}")
            self.medical_evaluator.print_medical_report(medical_metrics)
            
            if final_analysis:
                print(f"\n📊 GENERANDO VISUALIZACIONES PARA ARTÍCULO...")
                self.medical_evaluator.create_comprehensive_visualizations(medical_metrics, save_figs=True)
                
                # Exportar resultados
                csv_path = self.medical_evaluator.export_results_to_csv(
                    medical_metrics, 
                    f"skin_cancer_model_results_epoch_{epoch+1}.csv"
                )
                print(f"✅ Resultados exportados para artículo: {csv_path}")
        
        return avg_loss, accuracy, medical_metrics
    
    def train(self):
        print("🚀 Iniciando entrenamiento con análisis médico completo...")
        print("📊 Métricas médicas se generarán cada 5 épocas")
        print("🎯 Visualizaciones finales para artículo al completar")
        
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc, train_time = self.train_epoch(epoch)
            
            # Validate con análisis médico
            is_final = (epoch == self.config['epochs'] - 1)
            val_loss, val_acc, medical_metrics = self.validate_with_medical_analysis(epoch, final_analysis=is_final)
            
            # GPU stats
            gpu_stats = self.monitor_gpu()
            
            # Print clean stats
            print(f"\n{'='*80}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']}")
            print(f"{'='*80}")
            print(f"🔥 Entrenamiento:")
            print(f"   Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | Tiempo: {train_time:.1f}s")
            print(f"📊 Validación:")
            print(f"   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")
            
            # Métricas médicas resumidas
            if medical_metrics['melanoma']:
                mel = medical_metrics['melanoma']
                print(f"🩺 Melanoma: Sens {mel['melanoma_recall']:.1f}% | Prec {mel['melanoma_precision']:.1f}% | F1 {mel['melanoma_f1']:.1f}%")
            
            mb = medical_metrics['malignant_benign']
            print(f"⚡ Maligno: Sens {mb['malignant_sensitivity']:.1f}% | Espec {mb['malignant_specificity']:.1f}%")
            
            print(f"🖥️ Sistema: GPU {gpu_stats['gpu_util']:.0f}% | VRAM {gpu_stats['memory_gb']:.1f}GB | Temp {gpu_stats['temperature']:.0f}°C")
            
            # Score balanceado
            melanoma_recall = medical_metrics['melanoma']['melanoma_recall'] if medical_metrics['melanoma'] else 0
            balanced_score = 0.65 * val_acc + 0.35 * melanoma_recall
            
            # Save best models
            saved_model = False
            
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'medical_metrics': medical_metrics,
                    'config': self.config
                }, 'best_medical_model_accuracy.pth')
                print(f"✅ Mejor accuracy guardado: {val_acc:.2f}%")
                saved_model = True
            
            if medical_metrics['melanoma'] and medical_metrics['melanoma']['melanoma_recall'] > self.best_melanoma_recall:
                self.best_melanoma_recall = medical_metrics['melanoma']['melanoma_recall']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'melanoma_recall': self.best_melanoma_recall,
                    'medical_metrics': medical_metrics,
                    'config': self.config
                }, 'best_medical_model_melanoma.pth')
                print(f"✅ Mejor melanoma guardado: {self.best_melanoma_recall:.1f}%")
                saved_model = True
            
            if balanced_score > self.best_balanced_score:
                self.best_balanced_score = balanced_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'balanced_score': balanced_score,
                    'medical_metrics': medical_metrics,
                    'config': self.config
                }, 'best_medical_model_balanced.pth')
                print(f"✅ Mejor balance guardado: {balanced_score:.1f}%")
                saved_model = True
            
            if saved_model:
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                # Análisis final
                self.validate_with_medical_analysis(epoch, final_analysis=True)
                break
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"🏆 Mejor accuracy: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall:.1f}%")
        print(f"⚖️ Mejor score balanceado: {self.best_balanced_score:.1f}%")
        print(f"📊 Análisis médico completo disponible en: medical_analysis_figures/")

def main():
    """Entrenamiento RTX 3070 con análisis médico para artículo"""
    print("🏥 RTX 3070 Medical Training - Análisis completo para artículo")
    print("=" * 80)
    
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
    
    batch_size = 84
    
    print(f"Configuración:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Análisis médico: Completo con visualizaciones")
    
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
    print(f"   Clases: {list(label_encoder.classes_)}")
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',
        pretrained=True
    )
    
    # Entrenar con análisis médico
    trainer = RTX3070MedicalTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()