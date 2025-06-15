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
from model_phase3 import create_ciff_net_phase3, analyze_model_predictions

class CIFFNetPhase3Trainer:
    def __init__(self, model, train_loader, val_loader, label_encoder, device, config=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_encoder = label_encoder
        self.device = device
        
        # Configuración específica para Fase 3 (refinamiento)
        self.config = config or {
            'learning_rate': 1e-5,  # LR muy bajo para refinamiento
            'weight_decay': 1e-5,
            'epochs': 20,
            'early_stopping_patience': 6,
            'gradient_clipping': 0.3,
            'confidence_loss_weight': 0.1,  # Peso para loss de confianza
            'consistency_loss_weight': 0.05,  # Consistencia entre fases
        }
        
        self._setup_training()
        
        # Historial detallado
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.phase_accuracies = {'phase1': [], 'phase2': [], 'final': []}
        self.confidence_history = []
        self.prediction_change_history = []
        self.best_val_acc = 0.0
        
    def _setup_training(self):
        """Configurar optimizer específico para refinamiento"""
        
        # Solo entrenar el módulo de refinamiento final
        refinement_params = []
        for name, param in self.model.named_parameters():
            if 'final_refinement' in name and param.requires_grad:
                refinement_params.append(param)
        
        # Optimizer solo para refinamiento
        self.optimizer = optim.AdamW(
            refinement_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Loss functions
        self.main_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.BCELoss()  # Para calibración de confianza
        
        # Scheduler suave
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs'],
            eta_min=1e-7
        )
    
    def compute_confidence_targets(self, logits, labels):
        """Generar targets de confianza basados en correctness"""
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            confidence_targets = (predictions == labels).float().unsqueeze(1)
        return confidence_targets
    
    def compute_consistency_loss(self, phase1_logits, phase2_logits, final_logits):
        """Loss de consistencia para evitar cambios drásticos"""
        # KL divergence entre fases
        phase1_probs = torch.softmax(phase1_logits, dim=1)
        phase2_probs = torch.softmax(phase2_logits, dim=1)
        final_probs = torch.softmax(final_logits, dim=1)
        
        # Consistencia gradual
        kl_1_to_2 = nn.KLDivLoss(reduction='batchmean')(
            torch.log(phase2_probs + 1e-8), phase1_probs
        )
        kl_2_to_final = nn.KLDivLoss(reduction='batchmean')(
            torch.log(final_probs + 1e-8), phase2_probs
        )
        
        return (kl_1_to_2 + kl_2_to_final) / 2
    
    def train_epoch(self, epoch):
        """Entrenar época de refinamiento (Fase 3)"""
        self.model.train()
        
        running_main_loss = 0.0
        running_conf_loss = 0.0
        running_cons_loss = 0.0
        running_total_loss = 0.0
        
        correct_phase1 = correct_phase2 = correct_final = 0
        total = 0
        avg_confidence = 0
        
        pbar = tqdm(self.train_loader, desc=f'Phase3 Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            # Extraer datos
            main_images = batch['main_images'].to(self.device)
            context_images = batch['context_images'].to(self.device)
            main_labels = batch['main_labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass completo con análisis intermedio
            results = self.model(main_images, context_images, return_intermediate=True)
            
            phase1_logits = results['phase1_logits']
            phase2_logits = results['phase2_logits']
            final_logits = results['final_logits']
            confidence_score = results['confidence_score']
            
            # Loss principal (clasificación final)
            main_loss = self.main_criterion(final_logits, main_labels)
            
            # Loss de confianza
            confidence_targets = self.compute_confidence_targets(final_logits, main_labels)
            confidence_loss = self.confidence_criterion(confidence_score, confidence_targets)
            
            # Loss de consistencia
            consistency_loss = self.compute_consistency_loss(
                phase1_logits, phase2_logits, final_logits
            )
            
            # Loss total
            total_loss = (main_loss + 
                         self.config['confidence_loss_weight'] * confidence_loss +
                         self.config['consistency_loss_weight'] * consistency_loss)
            
            # Backward
            total_loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Estadísticas detalladas
            running_main_loss += main_loss.item()
            running_conf_loss += confidence_loss.item()
            running_cons_loss += consistency_loss.item()
            running_total_loss += total_loss.item()
            
            # Accuracies por fase
            _, pred1 = phase1_logits.max(1)
            _, pred2 = phase2_logits.max(1)
            _, pred_final = final_logits.max(1)
            
            total += main_labels.size(0)
            correct_phase1 += pred1.eq(main_labels).sum().item()
            correct_phase2 += pred2.eq(main_labels).sum().item()
            correct_final += pred_final.eq(main_labels).sum().item()
            
            avg_confidence += confidence_score.mean().item()
            
            # Actualizar barra
            pbar.set_postfix({
                'Main': f'{main_loss.item():.4f}',
                'Conf': f'{confidence_loss.item():.4f}',
                'Cons': f'{consistency_loss.item():.4f}',
                'Acc': f'{100.*correct_final/total:.2f}%',
                'ConfAvg': f'{confidence_score.mean().item():.3f}'
            })
        
        epoch_stats = {
            'main_loss': running_main_loss / len(self.train_loader),
            'confidence_loss': running_conf_loss / len(self.train_loader),
            'consistency_loss': running_cons_loss / len(self.train_loader),
            'total_loss': running_total_loss / len(self.train_loader),
            'phase1_acc': 100. * correct_phase1 / total,
            'phase2_acc': 100. * correct_phase2 / total,
            'final_acc': 100. * correct_final / total,
            'avg_confidence': avg_confidence / len(self.train_loader)
        }
        
        return epoch_stats
    
    def validate(self):
        """Validación completa con análisis por fases"""
        self.model.eval()
        running_loss = 0.0
        
        all_preds_phase1 = []
        all_preds_phase2 = []
        all_preds_final = []
        all_labels = []
        all_confidences = []
        prediction_changes = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating Phase3'):
                if isinstance(batch, dict):
                    images = batch['main_images'].to(self.device)
                    labels = batch['main_labels'].to(self.device)
                    context_images = batch.get('context_images', None)
                    if context_images is not None:
                        context_images = context_images.to(self.device)
                else:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    context_images = None
                
                # Análisis completo
                results = self.model(images, context_images, return_intermediate=True)
                final_logits = results['final_logits']
                confidence_score = results['confidence_score']
                
                loss = self.main_criterion(final_logits, labels)
                running_loss += loss.item()
                
                # Predicciones por fase
                _, pred1 = results['phase1_logits'].max(1)
                _, pred2 = results['phase2_logits'].max(1)
                _, pred_final = final_logits.max(1)
                
                all_preds_phase1.extend(pred1.cpu().numpy())
                all_preds_phase2.extend(pred2.cpu().numpy())
                all_preds_final.extend(pred_final.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence_score.cpu().numpy())
                
                # Cambios de predicción
                phase2_to_final_change = (pred2 != pred_final).float().mean().item()
                prediction_changes.append(phase2_to_final_change)
        
        val_loss = running_loss / len(self.val_loader)
        
        # Accuracies por fase
        acc_phase1 = accuracy_score(all_labels, all_preds_phase1) * 100
        acc_phase2 = accuracy_score(all_labels, all_preds_phase2) * 100
        acc_final = accuracy_score(all_labels, all_preds_final) * 100
        
        val_stats = {
            'val_loss': val_loss,
            'phase1_acc': acc_phase1,
            'phase2_acc': acc_phase2,
            'final_acc': acc_final,
            'avg_confidence': np.mean(all_confidences),
            'prediction_change_rate': np.mean(prediction_changes),
            'predictions': all_preds_final,
            'labels': all_labels
        }
        
        return val_stats
    
    def visualize_phase_evolution(self, sample_batch, epoch, save_path=None):
        """Visualizar evolución de predicciones a través de las fases"""
        self.model.eval()
        
        with torch.no_grad():
            main_images = sample_batch['main_images'][:6].to(self.device)
            context_images = sample_batch['context_images'][:6].to(self.device)
            labels = sample_batch['main_labels'][:6]
            
            analysis = self.model.get_phase_contributions(main_images, context_images)
            
            fig, axes = plt.subplots(2, 6, figsize=(18, 8))
            class_names = self.label_encoder.classes_
            
            for i in range(6):
                # Imagen
                img = main_images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Sample {i+1}\nTrue: {class_names[labels[i]]}')
                axes[0, i].axis('off')
                
                # Evolución de probabilidades
                phase1_probs = analysis['phase1_probs'][i].cpu().numpy()
                phase2_probs = analysis['phase2_probs'][i].cpu().numpy()
                final_probs = analysis['final_probs'][i].cpu().numpy()
                confidence = analysis['confidence'][i].item()
                
                x = np.arange(len(class_names))
                width = 0.25
                
                axes[1, i].bar(x - width, phase1_probs, width, label='Phase 1', alpha=0.7)
                axes[1, i].bar(x, phase2_probs, width, label='Phase 2', alpha=0.7)
                axes[1, i].bar(x + width, final_probs, width, label='Final', alpha=0.7)
                
                axes[1, i].set_title(f'Conf: {confidence:.3f}')
                axes[1, i].set_xlabel('Classes')
                axes[1, i].set_ylabel('Probability')
                axes[1, i].set_xticks(x)
                axes[1, i].set_xticklabels(class_names, rotation=45)
                if i == 0:
                    axes[1, i].legend()
            
            plt.suptitle(f'Phase Evolution Analysis - Epoch {epoch+1}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def train(self):
        """Entrenamiento completo de Fase 3"""
        print("🚀 Iniciando entrenamiento CIFF-Net Fase 3 - Refinamiento Final")
        print(f"📊 Configuración: {self.config}")
        print(f"🔧 Device: {self.device}")
        
        start_time = time.time()
        patience_counter = 0
        
        # Muestra para visualización
        sample_batch = next(iter(self.train_loader))
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch+1}/{self.config['epochs']} - CIFF-Net Fase 3")
            print(f"{'='*70}")
            
            # Entrenar
            train_stats = self.train_epoch(epoch)
            
            # Validar
            val_stats = self.validate()
            
            # Actualizar scheduler
            self.scheduler.step()
            
            # Guardar historial
            self.train_losses.append(train_stats['total_loss'])
            self.val_losses.append(val_stats['val_loss'])
            self.val_accuracies.append(val_stats['final_acc'])
            
            # Historial detallado
            self.phase_accuracies['phase1'].append(val_stats['phase1_acc'])
            self.phase_accuracies['phase2'].append(val_stats['phase2_acc'])
            self.phase_accuracies['final'].append(val_stats['final_acc'])
            
            self.confidence_history.append(val_stats['avg_confidence'])
            self.prediction_change_history.append(val_stats['prediction_change_rate'])
            
            # Mostrar resultados detallados
            print(f"🏋️  Train - Main: {train_stats['main_loss']:.4f}, "
                  f"Conf: {train_stats['confidence_loss']:.4f}, "
                  f"Cons: {train_stats['consistency_loss']:.4f}")
            print(f"📊 Train Acc - P1: {train_stats['phase1_acc']:.2f}%, "
                  f"P2: {train_stats['phase2_acc']:.2f}%, "
                  f"Final: {train_stats['final_acc']:.2f}%")
            print(f"✅ Val Acc - P1: {val_stats['phase1_acc']:.2f}%, "
                  f"P2: {val_stats['phase2_acc']:.2f}%, "
                  f"Final: {val_stats['final_acc']:.2f}%")
            print(f"🔍 Confidence: {val_stats['avg_confidence']:.3f}, "
                  f"Change Rate: {val_stats['prediction_change_rate']:.3f}")
            
            # Guardar mejor modelo
            if val_stats['final_acc'] > self.best_val_acc:
                self.best_val_acc = val_stats['final_acc']
                self.save_checkpoint(epoch, val_stats, 'best_ciff_net_phase3.pth')
                print(f"🎉 ¡Nuevo mejor modelo Fase 3! Acc: {val_stats['final_acc']:.2f}%")
                patience_counter = 0
                
                # Visualizar evolución
                if epoch % 3 == 0:
                    self.visualize_phase_evolution(
                        sample_batch, epoch,
                        f'phase_evolution_epoch_{epoch+1}.png'
                    )
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"⏹️  Early stopping en época {epoch+1}")
                break
        
        training_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total Fase 3: {training_time/3600:.2f} horas")
        
        # Resultados finales
        self.show_final_results(val_stats['predictions'], val_stats['labels'])
        self.plot_comprehensive_history()
    
    def save_checkpoint(self, epoch, val_stats, filename):
        """Guardar checkpoint completo de Fase 3"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_stats': val_stats,
            'best_val_acc': self.best_val_acc,
            'label_encoder': self.label_encoder,
            'config': self.config,
            'model_type': 'CIFF-Net-Phase3-Complete',
            'phase_accuracies': self.phase_accuracies,
            'confidence_history': self.confidence_history
        }, filename)
    
    def show_final_results(self, val_preds, val_labels):
        """Resultados finales completos de CIFF-Net"""
        class_names = self.label_encoder.classes_
        
        print(f"\n{'='*80}")
        print("🏆 RESULTADOS FINALES - CIFF-Net COMPLETO (3 FASES)")
        print(f"{'='*80}")
        
        # Reporte de clasificación
        print("\n📋 Reporte de Clasificación Final:")
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
            cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title('Matriz de Confusión - CIFF-Net Completo (3 Fases)',
                 fontsize=16, fontweight='bold')
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix_complete.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Resumen de mejoras por fase
        final_phase1_acc = self.phase_accuracies['phase1'][-1]
        final_phase2_acc = self.phase_accuracies['phase2'][-1]
        final_final_acc = self.phase_accuracies['final'][-1]
        
        print(f"\n📈 EVOLUCIÓN POR FASES:")
        print(f"  Fase 1 (MKSA): {final_phase1_acc:.2f}%")
        print(f"  Fase 2 (CCFF): {final_phase2_acc:.2f}% (+{final_phase2_acc-final_phase1_acc:.2f}%)")
        print(f"  Fase 3 (Refinamiento): {final_final_acc:.2f}% (+{final_final_acc-final_phase2_acc:.2f}%)")
        print(f"  🎯 Mejora total: +{final_final_acc-final_phase1_acc:.2f}%")
        
        print(f"\n🏆 Accuracy final: {self.best_val_acc:.2f}%")
        print(f"🔍 Confianza promedio: {self.confidence_history[-1]:.3f}")
    
    def plot_comprehensive_history(self):
        """Graficar historial completo de las 3 fases"""
        fig = plt.figure(figsize=(20, 12))
        
        # Configurar grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Loss evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.train_losses, label='Train Loss', color='blue', alpha=0.8)
        ax1.plot(self.val_losses, label='Val Loss', color='red', alpha=0.8)
        ax1.set_title('Loss Evolution', fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy por fases
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.phase_accuracies['phase1'], label='Fase 1 (MKSA)', color='orange')
        ax2.plot(self.phase_accuracies['phase2'], label='Fase 2 (CCFF)', color='green')
        ax2.plot(self.phase_accuracies['final'], label='Fase 3 (Final)', color='purple', linewidth=2)
        ax2.set_title('Accuracy Evolution por Fase', fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confidence evolution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.confidence_history, color='brown', linewidth=2)
        ax3.set_title('Confidence Score Evolution', fontweight='bold')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Avg Confidence')
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction change rate
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.plot(self.prediction_change_history, color='red', linewidth=2)
        ax4.set_title('Prediction Change Rate', fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Change Rate')
        ax4.grid(True, alpha=0.3)
        
        # 5. Comparación final
        ax5 = fig.add_subplot(gs[1, :2])
        phases = ['Fase 1\n(MKSA)', 'Fase 2\n(CCFF)', 'Fase 3\n(Refinamiento)']
        final_accs = [
            self.phase_accuracies['phase1'][-1],
            self.phase_accuracies['phase2'][-1],
            self.phase_accuracies['final'][-1]
        ]
        colors = ['orange', 'green', 'purple']
        bars = ax5.bar(phases, final_accs, color=colors, alpha=0.7)
        ax5.set_title('Accuracy Final por Fase', fontweight='bold', fontsize=14)
        ax5.set_ylabel('Accuracy (%)')
        
        # Añadir valores en barras
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Mejoras incrementales
        ax6 = fig.add_subplot(gs[1, 2:])
        improvements = [
            0,  # Baseline Fase 1
            final_accs[1] - final_accs[0],  # Mejora Fase 2
            final_accs[2] - final_accs[1]   # Mejora Fase 3
        ]
        cumulative = np.cumsum(improvements)
        
        ax6.bar(['Baseline\n(Fase 1)', 'Mejora\nFase 2', 'Mejora\nFase 3'], 
               improvements, color=['gray', 'lightgreen', 'lightpurple'], alpha=0.7)
        ax6.plot(['Baseline\n(Fase 1)', 'Mejora\nFase 2', 'Mejora\nFase 3'], 
                cumulative, 'ro-', linewidth=2, markersize=8)
        ax6.set_title('Mejoras Incrementales', fontweight='bold')
        ax6.set_ylabel('Mejora en Accuracy (%)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Resumen textual
        ax7 = fig.add_subplot(gs[2, :])
        summary_text = f"""
🎯 RESUMEN CIFF-NET COMPLETO:

📊 ARQUITECTURA:
• Fase 1: EfficientNet + Multi-Kernel Self-Attention (MKSA)
• Fase 2: Comparative Contextual Feature Fusion (CCFF)  
• Fase 3: Refinamiento Final con fusión de predicciones

📈 RESULTADOS FINALES:
• Accuracy Fase 1: {final_accs[0]:.2f}%
• Accuracy Fase 2: {final_accs[1]:.2f}% (+{improvements[1]:.2f}%)
• Accuracy Final: {final_accs[2]:.2f}% (+{improvements[2]:.2f}%)
• Mejora Total: +{sum(improvements[1:]):.2f}%
• Confianza Promedio: {self.confidence_history[-1]:.3f}

🔧 INNOVACIONES IMPLEMENTADAS:
✅ Multi-Kernel Self-Attention para enfoque en regiones relevantes
✅ Fusión contextual con múltiples imágenes relacionadas  
✅ Refinamiento final con calibración de confianza
✅ Análisis de consistencia entre fases
✅ Visualización de evolución de predicciones
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        ax7.axis('off')
        
        plt.suptitle('CIFF-Net - Historial Completo de Entrenamiento (3 Fases)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.savefig('ciff_net_complete_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Verificar RTX 3070 Ti
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Instala: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return
    
    device = torch.device('cuda')
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuración optimizada para RTX 3070 Ti
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'loss_type': 'focal',
        'scheduler': 'cosine',
        'epochs': 50,
        'early_stopping_patience': 12,
        'gradient_clipping': 1.0,
        'warmup_epochs': 5,
        'mixed_precision': True,  # Usar AMP para RTX 3070 Ti
    }
    
    # Batch size optimizado para 8GB VRAM
    batch_size = 28  # RTX 3070 Ti puede manejar esto bien
    
    # Cargar datos
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]
    
    print(f"📂 Cargando dataset (batch_size={batch_size})...")
    train_loader, val_loader, label_encoder = create_data_loaders(
        csv_file, image_folders, batch_size=batch_size
    )
    
    # Crear modelo RTX optimizado
    num_classes = len(label_encoder.classes_)
    print(f"🧠 Creando CIFF-Net RTX 3070 Ti para {num_classes} clases...")
    
    model = create_ciff_net_rtx3070ti(
        num_classes=num_classes,
        backbone='efficientnet_b2',  # Modelo más potente
        pretrained=True
    )
    
    # Resumen
    rtx_model_summary(model)
    
    # Entrenar con AMP
    trainer = CIFFNetPhase1TrainerRTX(
        model, train_loader, val_loader,
        label_encoder, device, training_config
    )
    
    trainer.train()

if __name__ == "__main__":
    main()