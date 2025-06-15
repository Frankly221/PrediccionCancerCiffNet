import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import time
import os
from datetime import datetime
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos
from phase1_feature_extraction import create_phase1_extractor
from phase2_cliff_detection_complete import create_phase2_complete_detector, CiffNetPhase2Loss
from phase3_classification_complete import create_phase3_complete_classifier, CiffNetPhase3Loss
from dataset_improved import create_improved_data_loaders
from metrics_and_visualization import CiffNetMetrics, create_all_visualizations

class CiffNetComplete(nn.Module):
    """
    Modelo CiffNet COMPLETO - Integración de las 3 fases
    Optimizado para RTX 3070 según paper original
    """
    
    def __init__(self, num_classes=7, cliff_threshold=0.15, backbone='efficientnet_b1'):
        super(CiffNetComplete, self).__init__()
        
        self.num_classes = num_classes
        self.cliff_threshold = cliff_threshold
        self.backbone = backbone
        
        print(f"🔧 CIFFNET COMPLETE - Inicializando modelo integrado...")
        print(f"   Backbone: {backbone}")
        print(f"   Classes: {num_classes}")
        print(f"   Cliff threshold: {cliff_threshold}")
        
        # ================================
        # FASE 1: Feature Extraction
        # ================================
        self.phase1 = create_phase1_extractor(
            backbone_name=backbone,
            pretrained=True
        )
        
        # ================================
        # FASE 2: Cliff Detection
        # ================================
        self.phase2 = create_phase2_complete_detector(
            input_dim=256,
            cliff_threshold=cliff_threshold,
            num_classes=num_classes
        )
        
        # ================================
        # FASE 3: Classification
        # ================================
        self.phase3 = create_phase3_complete_classifier(
            input_dim=256,
            num_classes=num_classes,
            cliff_threshold=cliff_threshold
        )
        
        # Loss functions
        self.phase2_loss = CiffNetPhase2Loss(alpha=1.0, beta=0.5, gamma=0.3, num_classes=num_classes)
        self.phase3_loss = CiffNetPhase3Loss(num_classes=num_classes, alpha=1.0, beta=0.3, gamma=0.2)
        
        print(f"✅ CiffNet Complete creado:")
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    
    def forward(self, images, targets=None, return_all=False):
        """
        Forward pass completo: Imagen → Predicción final
        """
        batch_size = images.size(0)
        
        # ================================
        # FASE 1: Feature Extraction
        # ================================
        with autocast():
            phase1_outputs = self.phase1(images)
            features = phase1_outputs['fused_features']  # [B, 256]
        
        # ================================
        # FASE 2: Cliff Detection
        # ================================
        with autocast():
            phase2_outputs = self.phase2(features)
            enhanced_features = phase2_outputs['enhanced_features']  # [B, 256]
        
        # ================================
        # FASE 3: Classification
        # ================================
        with autocast():
            phase3_outputs = self.phase3(phase2_outputs, return_all=return_all)
            final_predictions = phase3_outputs['predictions']  # [B]
            final_logits = phase3_outputs['logits']  # [B, num_classes]
            final_probs = phase3_outputs['probabilities']  # [B, num_classes]
        
        # Resultado principal
        result = {
            'logits': final_logits,
            'probabilities': final_probs,
            'predictions': final_predictions,
            'confidence': phase3_outputs['confidence'],
            'cliff_score': phase2_outputs['cliff_score'],
            'cliff_mask': phase2_outputs['cliff_mask']
        }
        
        # Información detallada si se requiere
        if return_all:
            result.update({
                'phase1_outputs': phase1_outputs,
                'phase2_outputs': phase2_outputs,
                'phase3_outputs': phase3_outputs
            })
        
        return result
    
    def compute_loss(self, outputs, targets):
        """
        Compute loss combinado de las 3 fases
        """
        # Extract outputs
        phase2_outputs = outputs.get('phase2_outputs', {})
        phase3_outputs = outputs.get('phase3_outputs', {})
        
        total_loss = 0.0
        loss_breakdown = {}
        
        # Phase 2 loss (si hay outputs disponibles)
        if phase2_outputs:
            try:
                phase2_loss_dict = self.phase2_loss(phase2_outputs, targets)
                phase2_total = phase2_loss_dict['total_loss']
                total_loss += 0.3 * phase2_total  # Peso 30% para fase 2
                loss_breakdown['phase2'] = phase2_loss_dict
            except Exception as e:
                print(f"⚠️ Warning computing Phase 2 loss: {e}")
        
        # Phase 3 loss (principal)
        if phase3_outputs:
            try:
                phase3_loss_dict = self.phase3_loss(phase3_outputs, targets)
                phase3_total = phase3_loss_dict['total_loss']
                total_loss += 1.0 * phase3_total  # Peso 100% para fase 3
                loss_breakdown['phase3'] = phase3_loss_dict
            except Exception as e:
                print(f"⚠️ Warning computing Phase 3 loss: {e}")
                # Fallback: usar CrossEntropy simple
                ce_loss = nn.CrossEntropyLoss()
                phase3_total = ce_loss(outputs['logits'], targets)
                total_loss += phase3_total
                loss_breakdown['phase3'] = {'total_loss': phase3_total}
        
        return total_loss, loss_breakdown

class CiffNetTrainer:
    """
    Entrenador optimizado para CiffNet con entrenamiento progresivo
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Optimización RTX 3070
        self.scaler = GradScaler() if config['mixed_precision'] else None
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['epochs'],
                eta_min=config['learning_rate'] * 0.01
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=10,
                verbose=True
            )
        
        # Métricas
        self.metrics_calc = CiffNetMetrics(
            num_classes=config['num_classes'],
            save_dir=config['save_dir']
        )
        
        # Historia de entrenamiento
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rate': []
        }
        
        # Best metrics tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
        print(f"✅ CiffNet Trainer inicializado:")
        print(f"   Optimizer: AdamW")
        print(f"   Scheduler: {config['scheduler']}")
        print(f"   Mixed precision: {config['mixed_precision']}")
        print(f"   Device: {device}")
    
    def train_epoch(self, epoch):
        """
        Entrenar una época
        """
        self.model.train()
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_targets = []
        all_predictions = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device, memory_format=torch.channels_last if self.config.get('channels_last', False) else torch.contiguous_format)
            targets = targets.to(self.device)
            
            # Forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(images, targets, return_all=True)
                    loss, _ = self.model.compute_loss(outputs, targets)
            else:
                outputs = self.model(images, targets, return_all=True)
                loss, _ = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    self.optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            predictions = outputs['predictions'].cpu()
            targets_cpu = targets.cpu()
            
            correct_predictions += (predictions == targets_cpu).sum().item()
            total_predictions += targets.size(0)
            
            all_targets.extend(targets_cpu.numpy())
            all_predictions.extend(predictions.numpy())
            
            # Update progress bar
            current_acc = correct_predictions / total_predictions
            pbar.set_postfix({
                'Loss': f"{running_loss/(batch_idx+1):.4f}",
                'Acc': f"{current_acc:.4f}"
            })
        
        # Métricas de época
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = correct_predictions / total_predictions
        
        # F1 score
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def validate_epoch(self, epoch):
        """
        Validar una época
        """
        self.model.eval()
        
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        all_targets = []
        all_predictions = []
        all_probs = []
        all_cliff_scores = []
        all_confidences = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device, memory_format=torch.channels_last if self.config.get('channels_last', False) else torch.contiguous_format)
                targets = targets.to(self.device)
                
                # Forward pass
                if self.scaler:
                    with autocast():
                        outputs = self.model(images, targets, return_all=True)
                        loss, _ = self.model.compute_loss(outputs, targets)
                else:
                    outputs = self.model(images, targets, return_all=True)
                    loss, _ = self.model.compute_loss(outputs, targets)
                
                # Estadísticas
                running_loss += loss.item()
                predictions = outputs['predictions'].cpu()
                probs = outputs['probabilities'].cpu()
                cliff_scores = outputs['cliff_score'].cpu()
                confidences = outputs['confidence'].cpu()
                targets_cpu = targets.cpu()
                
                correct_predictions += (predictions == targets_cpu).sum().item()
                total_predictions += targets.size(0)
                
                all_targets.extend(targets_cpu.numpy())
                all_predictions.extend(predictions.numpy())
                all_probs.append(probs.numpy())
                all_cliff_scores.extend(cliff_scores.numpy())
                all_confidences.extend(confidences.numpy())
        
        # Métricas de época
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = correct_predictions / total_predictions
        
        # Concatenar arrays
        all_probs = np.vstack(all_probs)
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_cliff_scores = np.array(all_cliff_scores).flatten()
        all_confidences = np.array(all_confidences).flatten()
        
        # F1 score
        from sklearn.metrics import f1_score
        epoch_f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Métricas detalladas en epoch final o mejores métricas
        if epoch >= self.config['epochs'] - 1 or epoch_f1 > self.best_val_f1:
            detailed_metrics = self._compute_detailed_metrics(
                all_targets, all_predictions, all_probs, 
                all_cliff_scores, all_confidences, epoch
            )
        else:
            detailed_metrics = None
        
        return epoch_loss, epoch_acc, epoch_f1, detailed_metrics
    
    def _compute_detailed_metrics(self, y_true, y_pred, y_probs, cliff_scores, confidences, epoch):
        """
        Compute métricas detalladas y generar visualizaciones
        """
        print(f"\n📊 Generando métricas detalladas para epoch {epoch+1}...")
        
        # Métricas básicas
        basic_metrics = self.metrics_calc.compute_basic_metrics(y_true, y_pred, y_probs)
        
        # Métricas cliff
        cliff_metrics = self.metrics_calc.compute_cliff_metrics(
            cliff_scores, None, y_true, y_pred, self.config['cliff_threshold']
        )
        
        # Métricas confianza
        confidence_metrics = self.metrics_calc.compute_confidence_metrics(
            confidences, y_true, y_pred
        )
        
        # Combinar métricas
        all_metrics = {
            'epoch': epoch + 1,
            'basic_metrics': basic_metrics,
            'cliff_metrics': cliff_metrics,
            'confidence_metrics': confidence_metrics
        }
        
        # Generar visualizaciones
        create_all_visualizations(
            self.metrics_calc, y_true, y_pred, y_probs,
            cliff_scores, confidences, self.history
        )
        
        # Guardar métricas
        self.metrics_calc.save_metrics_json(all_metrics, f"metrics_epoch_{epoch+1}.json")
        self.metrics_calc.generate_classification_report(y_true, y_pred, f"report_epoch_{epoch+1}.txt")
        
        return all_metrics
    
    def train(self):
        """
        Entrenamiento completo
        """
        print(f"🚀 Iniciando entrenamiento CiffNet Complete...")
        print(f"   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, val_f1, detailed_metrics = self.validate_epoch(epoch)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_f1)
            else:
                self.scheduler.step()
            
            # Save best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self._save_checkpoint(epoch, 'best')
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, f'epoch_{epoch+1}')
            
            epoch_time = time.time() - epoch_start
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['epochs']} Summary:")
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"   Time: {epoch_time:.1f}s, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if detailed_metrics:
                cliff_ratio = detailed_metrics['cliff_metrics']['cliff_ratio']
                cliff_perf = detailed_metrics['cliff_metrics']['performance_on_cliff']
                print(f"   Cliff: {cliff_ratio*100:.1f}% samples, {cliff_perf:.4f} accuracy on cliff")
        
        total_time = time.time() - start_time
        
        print(f"\n🎯 ENTRENAMIENTO COMPLETADO!")
        print(f"   Tiempo total: {total_time/60:.1f} minutos")
        print(f"   Mejor época: {self.best_epoch+1}")
        print(f"   Mejor Val Acc: {self.best_val_acc:.4f}")
        print(f"   Mejor Val F1: {self.best_val_f1:.4f}")
        
        # Guardar historia
        history_path = f"{self.config['save_dir']}/metrics/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        return self.history
    
    def _save_checkpoint(self, epoch, name):
        """
        Guardar checkpoint del modelo
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }
        
        save_path = f"{self.config['save_dir']}/models/ciffnet_{name}.pth"
        torch.save(checkpoint, save_path)
        print(f"💾 Checkpoint guardado: {save_path}")

def get_optimal_config():
    """
    Configuración óptima para RTX 3070
    """
    return {
        # Dataset
        'num_classes': 7,
        'batch_size': 32,
        'num_workers': 8,
        'pin_memory': True,
        
        # Training
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'gradient_accumulation_steps': 2,
        
        # Optimization
        'mixed_precision': True,
        'channels_last': True,
        'scheduler': 'cosine',
        
        # CiffNet specific
        'cliff_threshold': 0.15,
        'backbone': 'efficientnet_b1',
        
        # Output
        'save_dir': 'results',
        'save_every': 10
    }

def main():
    """
    Función principal - Orquestador completo
    """
    print("🎯 CIFFNET COMPLETE - PIPELINE COMPLETO")
    print("=" * 60)
    
    # Configuración
    config = get_optimal_config()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Crear directorios
    os.makedirs(config['save_dir'], exist_ok=True)
    for subdir in ['models', 'metrics', 'visualizations', 'analysis']:
        os.makedirs(f"{config['save_dir']}/{subdir}", exist_ok=True)
    
    # ================================
    # CARGAR DATASET
    # ================================
    print("\n📁 Cargando dataset HAM10000...")
    
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = [
        "datasetHam10000/HAM10000_images_part_1",
        "datasetHam10000/HAM10000_images_part_2"
    ]
    
    # LÍNEA CORREGIDA:
    train_loader, val_loader, label_encoder, class_weights = create_improved_data_loaders(
        csv_file=csv_file,
        image_folders=image_folders,
        batch_size=config['batch_size']
    )

    print(f"✅ Dataset cargado:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Classes: {label_encoder.classes_}")
    
    # ================================
    # CREAR MODELO
    # ================================
    print("\n🏗️ Creando modelo CiffNet Complete...")
    
    model = CiffNetComplete(
        num_classes=config['num_classes'],
        cliff_threshold=config['cliff_threshold'],
        backbone=config['backbone']
    )
    
    model = model.to(device)
    
    # Optimizaciones memoria
    if config['channels_last']:
        model = model.to(memory_format=torch.channels_last)
    
    print(f"✅ Modelo creado y movido a {device}")
    
    # ================================
    # ENTRENAMIENTO
    # ================================
    print("\n🏃 Iniciando entrenamiento...")
    
    trainer = CiffNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Entrenar modelo
    history = trainer.train()
    
    # ================================
    # EVALUACIÓN FINAL
    # ================================
    print("\n🧪 Evaluación final en conjunto de validación...")
    
    # Cargar mejor modelo
    best_model_path = f"{config['save_dir']}/models/ciffnet_best.pth"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Mejor modelo cargado desde epoch {checkpoint['epoch']+1}")
    
    # Evaluación final
    model.eval()
    all_targets = []
    all_predictions = []
    all_probs = []
    all_cliff_scores = []
    all_confidences = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluación final"):
            images = images.to(device, memory_format=torch.channels_last if config['channels_last'] else torch.contiguous_format)
            targets = targets.to(device)
            
            outputs = model(images, return_all=True)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs['predictions'].cpu().numpy())
            all_probs.append(outputs['probabilities'].cpu().numpy())
            all_cliff_scores.extend(outputs['cliff_score'].cpu().numpy())
            all_confidences.extend(outputs['confidence'].cpu().numpy())
    
    # Procesar resultados
    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)
    all_probs = np.vstack(all_probs)
    all_cliff_scores = np.array(all_cliff_scores).flatten()
    all_confidences = np.array(all_confidences).flatten()
    
    # ================================
    # MÉTRICAS FINALES
    # ================================
    print("\n📊 Generando métricas finales...")
    
    metrics_calc = CiffNetMetrics(num_classes=config['num_classes'], save_dir=config['save_dir'])
    
    # Métricas completas
    final_basic = metrics_calc.compute_basic_metrics(all_targets, all_predictions, all_probs)
    final_cliff = metrics_calc.compute_cliff_metrics(all_cliff_scores, None, all_targets, all_predictions)
    final_confidence = metrics_calc.compute_confidence_metrics(all_confidences, all_targets, all_predictions)
    
    final_metrics = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'basic_metrics': final_basic,
        'cliff_metrics': final_cliff,
        'confidence_metrics': final_confidence,
        'training_history': history
    }
    
    # Guardar métricas finales
    metrics_calc.save_metrics_json(final_metrics, "FINAL_METRICS.json")
    metrics_calc.generate_classification_report(all_targets, all_predictions, "FINAL_REPORT.txt")
    
    # Visualizaciones finales
    create_all_visualizations(
        metrics_calc, all_targets, all_predictions, all_probs,
        all_cliff_scores, all_confidences, history
    )
    
    # ================================
    # REPORTE FINAL
    # ================================
    print("\n" + "="*60)
    print("🎯 CIFFNET COMPLETE - RESULTADOS FINALES")
    print("="*60)
    print(f"📊 MÉTRICAS PRINCIPALES:")
    print(f"   Accuracy: {final_basic['accuracy']:.4f}")
    print(f"   F1-Score (macro): {final_basic['f1_macro']:.4f}")
    print(f"   F1-Score (weighted): {final_basic['f1_weighted']:.4f}")
    print(f"   AUC-ROC (macro): {final_basic.get('auc_macro', 'N/A')}")
    print(f"   Cohen's Kappa: {final_basic['cohen_kappa']:.4f}")
    print(f"   Matthews Correlation: {final_basic['matthews_corrcoef']:.4f}")
    
    print(f"\n🔬 CLIFF DETECTION:")
    print(f"   Cliff ratio: {final_cliff['cliff_ratio']*100:.1f}%")
    print(f"   Performance on cliff: {final_cliff['performance_on_cliff']:.4f}")
    print(f"   Performance on non-cliff: {final_cliff['performance_on_non_cliff']:.4f}")
    print(f"   Cliff detection F1: {final_cliff['cliff_detection_f1']:.4f}")
    
    print(f"\n📈 CONFIANZA & CALIBRACIÓN:")
    print(f"   Expected Calibration Error: {final_confidence['expected_calibration_error']:.4f}")
    print(f"   Confidence mean: {final_confidence['confidence_stats']['mean']:.4f}")
    print(f"   Confidence-accuracy correlation: {final_confidence['confidence_accuracy_correlation']:.4f}")
    
    print(f"\n💾 ARCHIVOS GENERADOS:")
    print(f"   📂 {config['save_dir']}/models/ - Modelos guardados")
    print(f"   📊 {config['save_dir']}/metrics/ - Métricas detalladas")
    print(f"   📈 {config['save_dir']}/visualizations/ - Gráficos (300 DPI)")
    print(f"   📄 {config['save_dir']}/analysis/ - Análisis adicionales")
    
    print(f"\n✅ PIPELINE COMPLETO FINALIZADO")
    print("="*60)

if __name__ == "__main__":
    # Parser para argumentos opcionales
    parser = argparse.ArgumentParser(description='CiffNet Complete Training Pipeline')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='results', help='Save directory')
    
    args = parser.parse_args()
    
    # Actualizar config con argumentos
    config = get_optimal_config()
    config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'save_dir': args.save_dir
    })
    
    # Ejecutar pipeline completo todo okey?
    main()