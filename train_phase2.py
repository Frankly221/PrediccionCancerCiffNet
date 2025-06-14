import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import time

from dataset import create_data_loaders_phase2
from model_phase2 import create_ciff_net_phase2

class CIFFNetPhase2Trainer:
    def __init__(self, model, train_loader, val_loader, label_encoder, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.device = device
        
        # Configuración específica para Fase 2
        self.config = config or {
            'learning_rate': 5e-5,  # LR más bajo para fine-tuning
            'weight_decay': 1e-4,
            'epochs': 30,
            'early_stopping_patience': 8,
            'gradient_clipping': 0.5,
            'warmup_epochs': 3,
            'context_loss_weight': 0.3,  # Peso para loss auxiliar de contexto
        }
        
        self._setup_training()
        
        # Historial
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.context_attention_history = []
        self.best_val_acc = 0.0
        
    def _setup_training(self):
        """Configurar optimizer y loss para Fase 2"""
        
        # Diferentes LR para diferentes partes
        phase1_params = []
        ccff_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'phase1_model' in name:
                    phase1_params.append(param)
                elif 'ccff_module' in name:
                    ccff_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Optimizer con LR diferenciados
        self.optimizer = optim.AdamW([
            {'params': phase1_params, 'lr': self.config['learning_rate'] * 0.1},  # Muy bajo para Fase 1
            {'params': ccff_params, 'lr': self.config['learning_rate']},          # Normal para CCFF
            {'params': classifier_params, 'lr': self.config['learning_rate'] * 2}  # Alto para clasificador
        ], weight_decay=self.config['weight_decay'])
        
        # Loss functions
        self.main_criterion = nn.CrossEntropyLoss()
        self.context_criterion = nn.CrossEntropyLoss()  # Loss auxiliar para contextos
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['epochs'], 
            eta_min=1e-7
        )
    
    def compute_context_auxiliary_loss(self, main_features, context_features, context_labels):
        """Loss auxiliar para que contextos sean informativos"""
        B, M, feature_dim = context_features.shape
        
        # Clasificar cada imagen contextual individualmente
        context_outputs = []
        for i in range(M):
            ctx_feat = context_features[:, i, :]  # [B, feature_dim]
            ctx_output = self.model.phase2_classifier(ctx_feat)  # [B, num_classes]
            context_outputs.append(ctx_output)
        
        # Loss promedio sobre contextos
        total_context_loss = 0
        for i, ctx_output in enumerate(context_outputs):
            ctx_labels = context_labels[:, i]
            total_context_loss += self.context_criterion(ctx_output, ctx_labels)
        
        return total_context_loss / M
    
    def train_epoch(self, epoch):
        """Entrenar época de Fase 2"""
        self.model.train()
        
        running_main_loss = 0.0
        running_context_loss = 0.0
        running_total_loss = 0.0
        correct = 0
        total = 0
        attention_weights_sum = 0
        
        pbar = tqdm(self.train_loader, desc=f'Phase2 Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Extraer datos del batch
            main_images = batch['main_images'].to(self.device)
            context_images = batch['context_images'].to(self.device)
            main_labels = batch['main_labels'].to(self.device)
            context_labels = batch['context_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, attention_weights = self.model(main_images, context_images)
            
            # Loss principal
            main_loss = self.main_criterion(output, main_labels)
            
            # Loss auxiliar de contexto (opcional)
            if self.config['context_loss_weight'] > 0:
                # Extraer características de contextos para loss auxiliar
                B, M = context_images.shape[:2]
                context_features_flat = context_images.view(B*M, *context_images.shape[2:])
                context_feat_extracted = self.model.extract_features(context_features_flat)
                context_features = context_feat_extracted.view(B, M, -1)
                
                context_loss = self.compute_context_auxiliary_loss(
                    None, context_features, context_labels
                )
            else:
                context_loss = torch.tensor(0.0, device=self.device)
            
            # Loss total
            total_loss = main_loss + self.config['context_loss_weight'] * context_loss
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Estadísticas
            running_main_loss += main_loss.item()
            running_context_loss += context_loss.item() if isinstance(context_loss, torch.Tensor) else context_loss
            running_total_loss += total_loss.item()
            
            _, predicted = output.max(1)
            total += main_labels.size(0)
            correct += predicted.eq(main_labels).sum().item()
            
            # Promediar pesos de atención para monitoreo
            if attention_weights is not None:
                attention_weights_sum += attention_weights.mean().item()
            
            # Actualizar barra
            pbar.set_postfix({
                'Main Loss': f'{main_loss.item():.4f}',
                'Ctx Loss': f'{context_loss.item() if isinstance(context_loss, torch.Tensor) else context_loss:.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'Att': f'{attention_weights.mean().item():.3f}' if attention_weights is not None else 'N/A'
            })
        
        epoch_main_loss = running_main_loss / len(self.train_loader)
        epoch_context_loss = running_context_loss / len(self.train_loader)
        epoch_total_loss = running_total_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        avg_attention = attention_weights_sum / len(self.train_loader)
        
        return epoch_main_loss, epoch_context_loss, epoch_total_loss, epoch_acc, avg_attention
    
    def validate(self):
        """Validación de Fase 2"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating Phase2'):
                if isinstance(batch, dict):
                    # Estructura compleja (no debería pasar en validación)
                    images = batch['main_images'].to(self.device)
                    labels = batch['main_labels'].to(self.device)
                else:
                    # Estructura simple
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                
                # Solo imagen principal en validación
                output, _ = self.model(images, None)
                loss = self.main_criterion(output, labels)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds) * 100
        
        return val_loss, val_acc, all_preds, all_labels
    
    def visualize_context_attention(self, sample_batch, epoch, save_path=None):
        """Visualizar atención contextual"""
        self.model.eval()
        
        with torch.no_grad():
            main_images = sample_batch['main_images'][:4].to(self.device)
            context_images = sample_batch['context_images'][:4].to(self.device)
            
            _, attention_weights = self.model(main_images, context_images)
            
            if attention_weights is not None:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                for i in range(4):
                    # Imagen principal
                    main_img = main_images[i].cpu().permute(1, 2, 0).numpy()
                    main_img = (main_img - main_img.min()) / (main_img.max() - main_img.min())
                    axes[0, i].imshow(main_img)
                    axes[0, i].set_title(f'Main Image {i+1}')
                    axes[0, i].axis('off')
                    
                    # Pesos de atención para contextos
                    att_weights = attention_weights[i].cpu().numpy()
                    axes[1, i].bar(range(len(att_weights)), att_weights)
                    axes[1, i].set_title(f'Context Attention {i+1}')
                    axes[1, i].set_xlabel('Context Image')
                    axes[1, i].set_ylabel('Attention Weight')
                
                plt.suptitle(f'Context Attention Visualization - Epoch {epoch+1}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
    
    def train(self):
        """Entrenamiento completo de Fase 2"""
        print("🚀 Iniciando entrenamiento CIFF-Net Fase 2 con CCFF")
        print(f"📊 Configuración: {self.config}")
        print(f"🔧 Device: {self.device}")
        
        start_time = time.time()
        patience_counter = 0
        
        # Muestra para visualización
        sample_batch = next(iter(self.train_loader))
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - CIFF-Net Fase 2")
            print(f"{'='*70}")
            
            # Entrenar
            main_loss, ctx_loss, total_loss, train_acc, avg_att = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Guardar historial
            self.train_losses.append(total_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.context_attention_history.append(avg_att)
            
            # Mostrar resultados
            print(f"🏋️  Train - Main: {main_loss:.4f}, Ctx: {ctx_loss:.4f}, Total: {total_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"✅ Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"👁️  Avg Context Attention: {avg_att:.3f}")
            print(f"📈 LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Guardar mejor modelo
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best_ciff_net_phase2.pth')
                print(f"🎉 ¡Nuevo mejor modelo Fase 2! Acc: {val_acc:.2f}%")
                patience_counter = 0
                
                # Visualizar atención
                if epoch % 5 == 0:
                    self.visualize_context_attention(
                        sample_batch, epoch, 
                        f'context_attention_epoch_{epoch+1}.png'
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️  Early stopping en época {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total Fase 2: {training_time/3600:.2f} horas")
        
        # Resultados finales
        self.show_final_results(val_preds, val_labels)
        self.plot_training_history()
    
    def save_checkpoint(self, epoch, val_acc, filename):
        """Guardar checkpoint de Fase 2"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'model_type': 'CIFF-Net-Phase2',
            'context_attention_history': self.context_attention_history
        }, filename)
    
    def show_final_results(self, val_preds, val_labels):
        """Resultados finales de Fase 2"""
        class_names = self.label_encoder.classes_
        
        print(f"\n{'='*80}")
        print("📊 RESULTADOS FINALES - CIFF-Net Fase 2 (CCFF)")
        print(f"{'='*80}")
        
        # Reporte de clasificación
        print("\n📋 Reporte de Clasificación:")
        report = classification_report(
            val_labels, val_preds, 
            target_names=class_names, 
            digits=4
        )
        print(report)
        
        # Matriz de confusión
        cm = confusion_matrix(val_labels, val_preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Matriz de Confusión - CIFF-Net Fase 2 (CCFF)', 
                 fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix_phase2.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n🏆 Mejor accuracy Fase 2: {self.best_val_acc:.2f}%")
    
    def plot_training_history(self):
        """Graficar historial específico de Fase 2"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Pérdida - CIFF-Net Fase 2')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='green')
        ax2.axhline(y=max(self.val_accuracies), color='red', linestyle='--', 
                   label=f'Best: {max(self.val_accuracies):.2f}%')
        ax2.set_title('Precisión - CIFF-Net Fase 2')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Context Attention
        ax3.plot(self.context_attention_history, color='purple', linewidth=2)
        ax3.set_title('Evolución Atención Contextual')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Peso Atención Promedio')
        ax3.grid(True, alpha=0.3)
        
        # Comparación fases (si disponible)
        ax4.text(0.1, 0.5, f'Fase 2 Completada\n\nMejor Accuracy: {self.best_val_acc:.2f}%\n\nMódulos:\n- MKSA (Fase 1)\n- CCFF (Fase 2)\n- Fusión Contextual', 
                transform=ax4.transAxes, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_title('Resumen Fase 2')
        ax4.axis('off')
        
        plt.suptitle('CIFF-Net Fase 2 - Historial de Entrenamiento', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history_phase2.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Configuración Fase 2
    training_config = {
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'epochs': 30,
        'early_stopping_patience': 8,
        'gradient_clipping': 0.5,
        'warmup_epochs': 3,
        'context_loss_weight': 0.2,
    }
    
    # Cargar datos para Fase 2 con rutas corregidas
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]  # Cambio aquí
    
    print("📂 Cargando dataset Fase 2...")
    train_loader, val_loader, label_encoder = create_data_loaders_phase2(
        csv_file, image_folders, 
        batch_size=8,  # Menor por usar múltiples imágenes
        num_context_images=3
    )
    
    # Crear modelo Fase 2
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net Fase 2...")
    
    try:
        model = create_ciff_net_phase2(
            phase1_model_path='best_ciff_net_phase1.pth',
            num_classes=num_classes,
            num_context_images=3,
            fusion_method='attention',
            freeze_phase1=False  # Permitir fine-tuning de Fase 1
        )
        
        # Entrenar
        trainer = CIFFNetPhase2Trainer(
            model, train_loader, val_loader,
            label_encoder, device, training_config
        )
        
        trainer.train()
        
        print("\n🎉 ¡Entrenamiento Fase 2 completado!")
        
    except FileNotFoundError:
        print("❌ Modelo de Fase 1 no encontrado.")
        print("💡 Entrena primero la Fase 1: python train.py")

if __name__ == "__main__":
    main()