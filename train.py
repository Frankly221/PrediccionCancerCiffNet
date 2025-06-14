import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from dataset import create_data_loaders
from model import create_ciff_net_phase1

class FocalLoss(nn.Module):
    """Focal Loss para datasets desbalanceados - usado en papers médicos"""
    def __init__(self, alpha=1, gamma=2, num_classes=7, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class LabelSmoothing(nn.Module):
    """Label Smoothing para mejorar generalización"""
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class CIFFNetPhase1Trainer:
    def __init__(self, model, train_loader, val_loader, label_encoder, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.device = device
        
        # Configuración por defecto
        self.config = config or {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'loss_type': 'focal',  # 'focal', 'cross_entropy', 'label_smooth'
            'scheduler': 'cosine',  # 'cosine', 'plateau'
            'epochs': 50,
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0,
            'warmup_epochs': 5
        }
        
        self._setup_training()
        
        # Historial
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.best_val_acc = 0.0
        
    def _setup_training(self):
        """Configurar optimizer, loss y scheduler para CIFF-Net"""
        
        # Diferentes learning rates para backbone y módulos MKSA
        backbone_params = []
        mksa_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'mksa' in name or 'attention' in name:
                mksa_params.append(param)
            else:
                classifier_params.append(param)
        
        # Optimizer con diferentes LR
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.1},
            {'params': mksa_params, 'lr': self.config['learning_rate'] * 0.5},
            {'params': classifier_params, 'lr': self.config['learning_rate']}
        ], weight_decay=self.config['weight_decay'])
        
        # Loss function
        if self.config['loss_type'] == 'focal':
            self.criterion = FocalLoss(alpha=1, gamma=2, num_classes=len(self.label_encoder.classes_))
        elif self.config['loss_type'] == 'label_smooth':
            self.criterion = LabelSmoothing(num_classes=len(self.label_encoder.classes_), smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        if self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs'], 
                eta_min=1e-6
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
    
    def warmup_lr(self, epoch):
        """Warmup del learning rate para estabilidad inicial"""
        if epoch < self.config['warmup_epochs']:
            warmup_factor = (epoch + 1) / self.config['warmup_epochs']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_factor
    
    def train_epoch(self, epoch):
        """Entrenar una época con progressive unfreezing para MKSA"""
        self.model.train()
        
        # Progressive unfreezing específico para CIFF-Net
        if epoch == 5:
            print("🔓 Unfreezing MKSA modules...")
            for param in self.model.mksa_modules.parameters():
                param.requires_grad = True
        elif epoch == 15:
            print("🔓 Unfreezing backbone...")
            self.model.freeze_backbone(False)
        
        # Warmup
        if epoch < self.config['warmup_epochs']:
            self.warmup_lr(epoch)
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'CIFF Phase1 Epoch {epoch+1}/{self.config["epochs"]}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping para estabilidad
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Actualizar barra
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validación del modelo CIFF-Net Fase 1"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validating CIFF Phase1'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds) * 100
        
        return val_loss, val_acc, all_preds, all_labels
    
    def visualize_attention(self, sample_images, save_path='attention_visualization.png'):
        """Visualizar mapas de atención del MKSA"""
        self.model.eval()
        
        with torch.no_grad():
            # Tomar una muestra pequeña
            sample_batch = sample_images[:4].to(self.device)
            attention_maps = self.model.get_attention_maps(sample_batch)
            
            fig, axes = plt.subplots(len(attention_maps), 4, figsize=(16, 4*len(attention_maps)))
            
            for level, att_maps in enumerate(attention_maps):
                for i in range(4):
                    # Promediar a través de canales para visualización
                    att_map = att_maps[i].mean(dim=0).cpu().numpy()
                    
                    axes[level, i].imshow(att_map, cmap='hot', interpolation='bilinear')
                    axes[level, i].set_title(f'Level {level+1}, Sample {i+1}')
                    axes[level, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def train(self):
        """Entrenamiento completo de CIFF-Net Fase 1"""
        print("🚀 Iniciando entrenamiento CIFF-Net Fase 1 con MKSA")
        print(f"📊 Configuración: {self.config}")
        print(f"🏷️  Clases: {list(self.label_encoder.classes_)}")
        print(f"🔧 Device: {self.device}")
        
        start_time = time.time()
        patience_counter = 0
        
        # Obtener muestra para visualización de atención
        sample_batch = next(iter(self.val_loader))[0]
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - CIFF-Net Fase 1")
            print(f"{'='*70}")
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Actualizar scheduler
            if self.config['scheduler'] == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_acc)
            
            # Guardar historial
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Mostrar resultados
            print(f"🏋️  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"✅ Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_ciff_net_phase1.pth')
                print(f"🎉 ¡Nuevo mejor modelo CIFF-Net Fase 1! Acc: {val_acc:.2f}%")
                patience_counter = 0
                
                # Visualizar atención cada vez que mejore
                if epoch % 5 == 0:
                    self.visualize_attention(sample_batch, f'attention_epoch_{epoch+1}.png')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️  Early stopping en época {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total de entrenamiento: {training_time/3600:.2f} horas")
        
        # Mostrar resultados finales
        self.show_final_results(val_preds, val_labels)
        self.plot_training_history()
        
        # Visualización final de atención
        self.visualize_attention(sample_batch, 'final_attention_maps.png')
    
    def save_checkpoint(self, epoch, val_acc, filename):
        """Guardar checkpoint del modelo CIFF-Net"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'model_type': 'CIFF-Net-Phase1'
        }, filename)
    
    def show_final_results(self, val_preds, val_labels):
        """Mostrar resultados finales con métricas médicas"""
        class_names = self.label_encoder.classes_
        
        print(f"\n{'='*80}")
        print("📊 RESULTADOS FINALES - CIFF-Net Fase 1")
        print(f"{'='*80}")
        
        # Reporte detallado
        print("\n📋 Reporte de Clasificación:")
        report = classification_report(
            val_labels, val_preds, 
            target_names=class_names, 
            digits=4
        )
        print(report)
        
        # Matriz de confusión mejorada
        cm = confusion_matrix(val_labels, val_preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Número de muestras'}
        )
        plt.title('Matriz de Confusión - CIFF-Net Fase 1 con MKSA', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Verdadera', fontsize=12)
        plt.xlabel('Etiqueta Predicha', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_ciff_phase1.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Métricas por clase
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        print(f"\n🎯 Accuracy por clase:")
        for i, (class_name, acc) in enumerate(zip(class_names, class_accuracies)):
            print(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)")
            
        print(f"\n🏆 Mejor accuracy alcanzada: {self.best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """Graficar historial de entrenamiento específico para CIFF-Net"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.8)
        ax1.plot(self.val_losses, label='Val Loss', color='red', alpha=0.8)
        ax1.set_title('Pérdida - CIFF-Net Fase 1', fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green', linewidth=2)
        ax2.axhline(y=max(self.val_accuracies), color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: {max(self.val_accuracies):.2f}%')
        ax2.set_title('Precisión - CIFF-Net Fase 1', fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning Rate
        ax3.plot(self.learning_rates, color='orange', linewidth=2)
        ax3.set_title('Learning Rate Schedule', fontweight='bold')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Convergencia
        smoothed_acc = np.convolve(self.val_accuracies, np.ones(5)/5, mode='valid')
        ax4.plot(range(len(smoothed_acc)), smoothed_acc, color='purple', linewidth=2)
        ax4.set_title('Convergencia (Suavizada)', fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('CIFF-Net Fase 1 - Historial de Entrenamiento', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history_ciff_phase1.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Configuración
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")
    
    # Configuración específica para CIFF-Net Fase 1
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'loss_type': 'focal',  # Ideal para datos médicos desbalanceados
        'scheduler': 'cosine',
        'epochs': 50,
        'early_stopping_patience': 12,
        'gradient_clipping': 1.0,
        'warmup_epochs': 5
    }
    
    # Cargar datos con rutas corregidas
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]  # Cambio aquí
    
    print("📂 Cargando dataset HAM10000 para CIFF-Net...")
    train_loader, val_loader, label_encoder = create_data_loaders(
        csv_file, image_folders, batch_size=24  # Reduce si hay problemas de VRAM
    )
    
    # Crear modelo CIFF-Net Fase 1
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net Fase 1 con MKSA para {num_classes} clases...")
    
    model = create_ciff_net_phase1(
        num_classes=num_classes, 
        backbone='efficientnet_b0',  # Cambiar a 'efficientnet_b3' si tienes más VRAM
        pretrained=True
    )
    
    # Resumen del modelo
    from model import model_summary
    model_summary(model)
    
    # Entrenar
    trainer = CIFFNetPhase1Trainer(
        model, train_loader, val_loader, 
        label_encoder, device, training_config
    )
    
    trainer.train()
    
    print("\n🎉 ¡Entrenamiento de CIFF-Net Fase 1 completado!")
    print("📁 Archivos generados:")
    print("  - best_ciff_net_phase1.pth (modelo entrenado)")
    print("  - confusion_matrix_ciff_phase1.png")
    print("  - training_history_ciff_phase1.png")
    print("  - attention_maps visualizaciones")

if __name__ == "__main__":
    main()