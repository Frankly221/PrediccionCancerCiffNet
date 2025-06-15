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

from dataset_improved import create_improved_data_loaders
from model_improved import create_improved_ciff_net, FocalLoss

class ImprovedCIFFNetTrainer:
    """Trainer mejorado con técnicas avanzadas"""
    def __init__(self, model, train_loader, val_loader, label_encoder, class_weights, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.label_encoder = label_encoder
        self.class_weights = class_weights.to(device)
        
        # AMP
        self.scaler = GradScaler()
        
        # Optimizer mejorado
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler adaptativo - ARREGLADO
        if config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='max',
                factor=0.5,
                patience=5,
                min_lr=1e-7
                # verbose=True  ← REMOVIDO (no existe en PyTorch)
            )
            self.scheduler_verbose = True  # Flag manual para prints
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-7
            )
            self.scheduler_verbose = False
        
        # Loss mejorado
        if config['loss_type'] == 'focal':
            self.criterion = FocalLoss(
                alpha=1.0,
                gamma=2.0,
                weight=self.class_weights
            )
        elif config['loss_type'] == 'weighted_ce':
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.melanoma_recalls = []  # Específico para melanoma
        self.learning_rates = []    # Track LR changes
        
        # Best metrics tracking
        self.best_overall_acc = 0
        self.best_melanoma_recall = 0
        self.best_balanced_score = 0
        
        print(f"🚀 Trainer MEJORADO inicializado:")
        print(f"   Optimizer: AdamW")
        print(f"   Scheduler: {config['scheduler']}")
        print(f"   Loss: {config['loss_type']}")
        print(f"   LR inicial: {config['learning_rate']}")
        
    def compute_melanoma_metrics(self, all_predicted, all_targets):
        """Calcular métricas específicas para melanoma"""
        # Encontrar índice de melanoma
        melanoma_idx = None
        for i, class_name in enumerate(self.label_encoder.classes_):
            if 'mel' in class_name.lower():
                melanoma_idx = i
                break
        
        if melanoma_idx is None:
            return 0.0, 0.0
        
        # Crear máscaras binarias para melanoma
        melanoma_true = (np.array(all_targets) == melanoma_idx)
        melanoma_pred = (np.array(all_predicted) == melanoma_idx)
        
        # Calcular recall y precision
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
        
        pbar = tqdm(self.train_loader, desc=f"🔥 Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward con AMP
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
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
            
            # Update progress
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'VRAM': f"{torch.cuda.memory_allocated()/1e9:.1f}GB",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Memory cleanup
            if batch_idx % 30 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predicted = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="🔍 Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Probabilities para ROC-AUC
                probs = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probs.cpu().numpy())
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Métricas específicas para melanoma
        melanoma_recall, melanoma_precision = self.compute_melanoma_metrics(all_predicted, all_targets)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.melanoma_recalls.append(melanoma_recall * 100)
        
        return avg_loss, accuracy, all_predicted, all_targets, all_probabilities, melanoma_recall, melanoma_precision
    
    def compute_balanced_score(self, accuracy, melanoma_recall):
        """Score balanceado que prioriza melanoma"""
        # Peso mayor para melanoma (crítico médicamente)
        return 0.6 * accuracy + 0.4 * (melanoma_recall * 100)
    
    def save_enhanced_plots(self):
        """Guardar gráficos mejorados"""
        plt.figure(figsize=(20, 12))
        
        # Loss plot
        plt.subplot(2, 4, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', color='red', linewidth=2)
        plt.title('Loss Durante Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(2, 4, 2)
        plt.plot(self.train_accuracies, label='Train Acc', color='blue', linewidth=2)
        plt.plot(self.val_accuracies, label='Val Acc', color='red', linewidth=2)
        plt.title('Accuracy Durante Entrenamiento', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Melanoma recall plot
        plt.subplot(2, 4, 3)
        plt.plot(self.melanoma_recalls, label='Melanoma Recall', color='red', linewidth=2)
        plt.title('Melanoma Recall (Crítico)', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Recall (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(2, 4, 4)
        plt.plot(self.learning_rates, color='green', linewidth=2)
        plt.title('Learning Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Comparación de métricas
        plt.subplot(2, 4, 5)
        epochs = range(1, len(self.val_accuracies) + 1)
        plt.plot(epochs, self.val_accuracies, label='Overall Acc', linewidth=2)
        plt.plot(epochs, self.melanoma_recalls, label='Melanoma Recall', linewidth=2)
        balanced_scores = [self.compute_balanced_score(acc, mel_rec/100) for acc, mel_rec in zip(self.val_accuracies, self.melanoma_recalls)]
        plt.plot(epochs, balanced_scores, label='Balanced Score', linewidth=2, linestyle='--')
        plt.title('Métricas Comparativas', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Score (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Overfitting check
        plt.subplot(2, 4, 6)
        gap = [train - val for train, val in zip(self.train_accuracies, self.val_accuracies)]
        plt.plot(gap, color='purple', linewidth=2)
        plt.title('Train-Val Gap (Overfitting Check)', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Gap (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='Warning')
        plt.axhline(y=10, color='red', linestyle='-', alpha=0.7, label='Overfitting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_enhanced_confusion_matrix(self, all_predicted, all_targets):
        """Matriz de confusión mejorada"""
        plt.figure(figsize=(12, 10))
        
        cm = confusion_matrix(all_targets, all_predicted)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Heatmap mejorado
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   cbar_kws={'label': 'Accuracy Normalizada'},
                   square=True)
        
        plt.title('Matriz de Confusión - CIFF-Net Mejorado', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicción', fontsize=14)
        plt.ylabel('Real', fontsize=14)
        
        # Destacar melanoma
        melanoma_idx = None
        for i, class_name in enumerate(self.label_encoder.classes_):
            if 'mel' in class_name.lower():
                melanoma_idx = i
                break
        
        if melanoma_idx is not None:
            # Resaltar fila de melanoma
            plt.axhline(y=melanoma_idx, color='red', linewidth=3, alpha=0.7)
            plt.axhline(y=melanoma_idx+1, color='red', linewidth=3, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Reporte de clasificación
        report = classification_report(
            all_targets, all_predicted,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv('classification_report_improved.csv')
        
        return report
    
    def train(self):
        print(f"🚀 Iniciando entrenamiento CIFF-Net MEJORADO...")
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
            val_loss, val_acc, all_predicted, all_targets, all_probs, melanoma_recall, melanoma_precision = self.validate()
            
            # Scheduler step con verbose manual
            current_lr = self.optimizer.param_groups[0]['lr']
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_acc)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                # Manual verbose para ReduceLROnPlateau
                if self.scheduler_verbose and new_lr < previous_lr:
                    print(f"📉 ReduceLROnPlateau: reducing learning rate to {new_lr:.2e}")
                previous_lr = new_lr
            else:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Compute balanced score
            balanced_score = self.compute_balanced_score(val_acc, melanoma_recall)
            
            # Print enhanced stats
            print(f"\n{'='*80}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - CIFF-Net MEJORADO")
            print(f"{'='*80}")
            print(f"🔥 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"🩺 Melanoma Recall: {melanoma_recall*100:.2f}% | Precision: {melanoma_precision*100:.2f}%")
            print(f"⚖️  Balanced Score: {balanced_score:.2f}%")
            print(f"⏱️  Tiempo época: {epoch_time:.1f}s")
            print(f"💾 VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
            print(f"📈 LR actual: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best models con múltiples criterios
            saved_model = False
            
            # Mejor accuracy general
            if val_acc > self.best_overall_acc:
                self.best_overall_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'config': self.config
                }, 'best_overall_improved.pth')
                print(f"✅ Mejor accuracy general guardado: {val_acc:.2f}%")
                saved_model = True
            
            # Mejor melanoma recall
            if melanoma_recall > self.best_melanoma_recall:
                self.best_melanoma_recall = melanoma_recall
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_melanoma_recall': melanoma_recall,
                    'val_acc': val_acc,
                    'config': self.config
                }, 'best_melanoma_improved.pth')
                print(f"✅ Mejor melanoma recall guardado: {melanoma_recall*100:.2f}%")
                saved_model = True
            
            # Mejor score balanceado
            if balanced_score > self.best_balanced_score:
                self.best_balanced_score = balanced_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'balanced_score': balanced_score,
                    'val_acc': val_acc,
                    'melanoma_recall': melanoma_recall,
                    'config': self.config
                }, 'best_balanced_improved.pth')
                print(f"✅ Mejor score balanceado guardado: {balanced_score:.2f}%")
                saved_model = True
            
            if saved_model:
                patience_counter = 0
                # Guardar métricas del mejor modelo
                self.save_enhanced_confusion_matrix(all_predicted, all_targets)
            else:
                patience_counter += 1
                print(f"⏳ Paciencia: {patience_counter}/{self.config['early_stopping_patience']}")
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️ Early stopping en época {epoch+1}")
                break
            
            # Guardar plots cada 5 épocas
            if (epoch + 1) % 5 == 0:
                self.save_enhanced_plots()
        
        total_time = time.time() - start_time
        self.save_enhanced_plots()
        
        print(f"\n🎉 Entrenamiento MEJORADO completado!")
        print(f"⏱️  Tiempo total: {total_time/3600:.2f} horas")
        print(f"🏆 Mejor accuracy general: {self.best_overall_acc:.2f}%")
        print(f"🩺 Mejor melanoma recall: {self.best_melanoma_recall*100:.2f}%")
        print(f"⚖️  Mejor score balanceado: {self.best_balanced_score:.2f}%")

def main():
    # Configuración mejorada
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Dispositivo: {device}")
    
    # Configuración optimizada
    training_config = {
        'learning_rate': 1e-4,  # LR más conservador
        'weight_decay': 2e-4,   # Más regularización
        'loss_type': 'focal',   # Focal loss para desbalance
        'scheduler': 'plateau', # Scheduler adaptativo
        'epochs': 60,          # Más épocas
        'early_stopping_patience': 15,  # Más paciencia
        'gradient_clipping': 0.5,       # Menos clipping
        'mixed_precision': True
    }
    
    batch_size = 12  # Reducido para modelo más grande
    
    # Cargar datos mejorados
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    print(f"📂 Cargando dataset mejorado...")
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file, image_folders, batch_size=batch_size
    )
    
    # Crear modelo mejorado
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net MEJORADO para {num_classes} clases...")
    
    model = create_improved_ciff_net(
        num_classes=num_classes,
        backbone='efficientnet_b1',  # Modelo más potente
        pretrained=True
    )
    
    # Entrenar modelo mejorado
    trainer = ImprovedCIFFNetTrainer(
        model, train_loader, val_loader,
        label_encoder, class_weights, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()