import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import time
import warnings
warnings.filterwarnings('ignore')

from dataset import create_data_loaders, HAM10000Dataset
from model import create_ciff_net_rtx8gb, rtx8gb_model_summary

class CIFFNetPhase1TrainerAMP:
    """Trainer con AMP para RTX 8GB"""
    def __init__(self, model, train_loader, val_loader, label_encoder, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        
        # Configurar AMP
        self.scaler = GradScaler()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # Loss con diferentes opciones
        if config.get('loss_type') == 'focal':
            # Focal Loss para desbalance de clases
            from torch.nn import CrossEntropyLoss
            self.criterion = CrossEntropyLoss(label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Para tracking del entrenamiento
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"RTX Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward con autocast (AMP)
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward con scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if 'gradient_clipping' in self.config:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'VRAM': f"{torch.cuda.memory_allocated()/1e9:.1f}GB",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Limpiar cache cada cierto número de batches
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Para métricas detalladas
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy, all_predicted, all_targets
    
    def save_plots(self):
        """Guardar gráficos de entrenamiento"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.title('Loss Durante Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='blue')
        plt.plot(self.val_accuracies, label='Val Acc', color='red')
        plt.title('Accuracy Durante Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Learning rate plot
        plt.subplot(1, 3, 3)
        if len(self.train_losses) > 0:
            lrs = []
            for i in range(len(self.train_losses)):
                lrs.append(self.config['learning_rate'] * (0.5 ** (i // 10)))  # Aproximación
            plt.plot(lrs, color='green')
            plt.title('Learning Rate')
            plt.xlabel('Época')
            plt.ylabel('LR')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_rtx8gb.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_confusion_matrix(self, all_predicted, all_targets):
        """Guardar matriz de confusión"""
        plt.figure(figsize=(10, 8))
        
        # Matriz de confusión
        cm = confusion_matrix(all_targets, all_predicted)
        
        # Normalizar
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        
        plt.title('Matriz de Confusión Normalizada - CIFF-Net RTX')
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig('confusion_matrix_rtx8gb.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reporte de clasificación
        report = classification_report(
            all_targets, all_predicted,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        # Guardar reporte
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report_rtx8gb.csv')
        
        return report
    
    def train(self):
        print(f"🚀 Iniciando entrenamiento CIFF-Net RTX 8GB con AMP...")
        print(f"📊 Configuración: {self.config}")
        
        best_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Limpiar cache al inicio de cada época
            torch.cuda.empty_cache()
            
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, all_predicted, all_targets = self.validate()
            
            # Scheduler step
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Print stats
            print(f"\n{'='*60}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - CIFF-Net RTX")
            print(f"{'='*60}")
            print(f"🔥 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"⏱️  Tiempo época: {epoch_time:.1f}s")
            print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
            print(f"📈 LR actual: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                
                # Guardar modelo
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_acc': best_acc,
                    'config': self.config
                }, 'best_ciff_net_rtx8gb.pth')
                
                print(f"✅ Nuevo mejor modelo guardado! Acc: {best_acc:.2f}%")
                
                # Guardar métricas del mejor modelo
                best_report = self.save_confusion_matrix(all_predicted, all_targets)
                
            else:
                patience_counter += 1
                print(f"⏳ Paciencia: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                break
            
            # Guardar plots cada 5 épocas
            if (epoch + 1) % 5 == 0:
                self.save_plots()
        
        total_time = time.time() - start_time
        
        # Guardar plots finales
        self.save_plots()
        
        print(f"\n🎉 Entrenamiento completado!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy: {best_acc:.2f}%")
        print(f"📁 Archivos generados:")
        print(f"   - best_ciff_net_rtx8gb.pth")
        print(f"   - training_history_rtx8gb.png")
        print(f"   - confusion_matrix_rtx8gb.png") 
        print(f"   - classification_report_rtx8gb.csv")

def main():
    # Limpiar cache CUDA al inicio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Configurar para evitar fragmentación
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuración optimizada para RTX 8GB
    training_config = {
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'loss_type': 'focal',
        'scheduler': 'cosine',
        'epochs': 40,
        'early_stopping_patience': 10,
        'gradient_clipping': 1.0,
        'warmup_epochs': 3,
        'mixed_precision': True
    }
    
    # Batch size para RTX 8GB
    batch_size = 16
    
    # Cargar datos
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    print(f"📂 Cargando dataset HAM10000 para CIFF-Net...")
    train_loader, val_loader, label_encoder = create_data_loaders(
        csv_file, image_folders, batch_size=batch_size
    )
    
    # Crear modelo
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net RTX 8GB para {num_classes} clases...")
    
    model = create_ciff_net_rtx8gb(
        num_classes=num_classes,
        backbone='efficientnet_b0',
        pretrained=True
    )
    
    # Resumen del modelo
    rtx8gb_model_summary(model)
    
    # Entrenar con AMP
    trainer = CIFFNetPhase1TrainerAMP(
        model, train_loader, val_loader,
        label_encoder, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()