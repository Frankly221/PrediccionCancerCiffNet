import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import create_ciff_net_phase1
from model_phase2 import create_ciff_net_phase2

class FinalRefinementModule(nn.Module):
    """Módulo de refinamiento final que fusiona Fase 1 + Fase 2"""
    def __init__(self, num_classes=7, fusion_method='weighted_concatenation'):
        super(FinalRefinementModule, self).__init__()
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        if fusion_method == 'weighted_concatenation':
            # Concatenar logits de ambas fases + capas FC
            self.fusion_layers = nn.Sequential(
                nn.Linear(num_classes * 2, 128),  # Logits fase1 + fase2
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
            
            # Pesos aprendibles para cada fase
            self.phase_weights = nn.Parameter(torch.tensor([0.4, 0.6]))  # [fase1, fase2]
            
        elif fusion_method == 'attention_fusion':
            # Atención entre predicciones de fases
            self.attention_layer = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),  # Peso para cada fase
                nn.Softmax(dim=1)
            )
            
            self.refinement_layers = nn.Sequential(
                nn.Linear(num_classes, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, num_classes)
            )
            
        elif fusion_method == 'ensemble_voting':
            # Votación ponderada simple
            self.voting_weights = nn.Parameter(torch.tensor([0.3, 0.7]))
            
        # Calibración de confianza
        self.confidence_calibration = nn.Sequential(
            nn.Linear(num_classes, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Inicializar pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos del módulo de refinamiento"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, phase1_logits, phase2_logits):
        """
        phase1_logits: [B, num_classes] - Predicciones de Fase 1
        phase2_logits: [B, num_classes] - Predicciones de Fase 2
        """
        
        if self.fusion_method == 'weighted_concatenation':
            # Normalizar pesos de fases
            normalized_weights = F.softmax(self.phase_weights, dim=0)
            
            # Ponderar logits
            weighted_phase1 = phase1_logits * normalized_weights[0]
            weighted_phase2 = phase2_logits * normalized_weights[1]
            
            # Concatenar para fusión
            concatenated = torch.cat([weighted_phase1, weighted_phase2], dim=1)
            final_logits = self.fusion_layers(concatenated)
            
        elif self.fusion_method == 'attention_fusion':
            # Concatenar para calcular atención
            concatenated = torch.cat([phase1_logits, phase2_logits], dim=1)
            attention_weights = self.attention_layer(concatenated)  # [B, 2]
            
            # Aplicar atención
            attended_phase1 = phase1_logits * attention_weights[:, 0:1]
            attended_phase2 = phase2_logits * attention_weights[:, 1:2]
            
            # Fusión final
            fused = attended_phase1 + attended_phase2
            final_logits = self.refinement_layers(fused)
            
        elif self.fusion_method == 'ensemble_voting':
            # Votación ponderada simple
            normalized_weights = F.softmax(self.voting_weights, dim=0)
            final_logits = (phase1_logits * normalized_weights[0] + 
                           phase2_logits * normalized_weights[1])
        
        # Calibración de confianza
        confidence_score = self.confidence_calibration(final_logits)
        
        return final_logits, confidence_score

class CIFFNetPhase3(nn.Module):
    """CIFF-Net Fase 3 Completo: Fase 1 + Fase 2 + Refinamiento Final"""
    def __init__(self, phase1_model_path, phase2_model_path, num_classes=7, 
                 num_context_images=3, fusion_method='weighted_concatenation',
                 freeze_previous_phases=True):
        super(CIFFNetPhase3, self).__init__()
        
        # Cargar modelos pre-entrenados de Fase 1 y Fase 2
        print("📂 Cargando modelo de Fase 1...")
        checkpoint1 = torch.load(phase1_model_path, map_location='cpu')
        self.phase1_model = create_ciff_net_phase1(num_classes=num_classes)
        self.phase1_model.load_state_dict(checkpoint1['model_state_dict'])
        
        print("📂 Cargando modelo de Fase 2...")
        checkpoint2 = torch.load(phase2_model_path, map_location='cpu')
        self.phase2_model = create_ciff_net_phase2(
            phase1_model_path, num_classes, num_context_images, 'attention'
        )
        self.phase2_model.load_state_dict(checkpoint2['model_state_dict'])
        
        # Congelar fases anteriores si se especifica
        if freeze_previous_phases:
            for param in self.phase1_model.parameters():
                param.requires_grad = False
            for param in self.phase2_model.parameters():
                param.requires_grad = False
        
        # Módulo de refinamiento final
        self.final_refinement = FinalRefinementModule(
            num_classes=num_classes,
            fusion_method=fusion_method
        )
        
        self.num_context_images = num_context_images
        self.num_classes = num_classes
        
        print(f"✅ CIFF-Net Fase 3 inicializado con método: {fusion_method}")
    
    def forward(self, main_images, context_images=None, return_intermediate=False):
        """
        Forward pass completo de las 3 fases
        
        main_images: [B, C, H, W] - Imágenes principales
        context_images: [B, M, C, H, W] - Imágenes contextuales (para Fase 2)
        return_intermediate: Si retornar predicciones intermedias
        """
        
        # FASE 1: Clasificación con MKSA
        self.phase1_model.eval()
        with torch.set_grad_enabled(self.training and not self.freeze_previous_phases):
            phase1_logits = self.phase1_model(main_images)
        
        # FASE 2: Fusión contextual con CCFF
        self.phase2_model.eval()
        with torch.set_grad_enabled(self.training and not self.freeze_previous_phases):
            if context_images is not None and self.training:
                phase2_logits, attention_weights = self.phase2_model(main_images, context_images)
            else:
                # Solo imagen principal en validación/inferencia
                phase2_logits, attention_weights = self.phase2_model(main_images, None)
        
        # FASE 3: Refinamiento final
        final_logits, confidence_score = self.final_refinement(phase1_logits, phase2_logits)
        
        if return_intermediate:
            return {
                'phase1_logits': phase1_logits,
                'phase2_logits': phase2_logits,
                'final_logits': final_logits,
                'confidence_score': confidence_score,
                'attention_weights': attention_weights
            }
        else:
            return final_logits, confidence_score
    
    def get_phase_contributions(self, main_images, context_images=None):
        """Analizar contribuciones de cada fase"""
        self.eval()
        with torch.no_grad():
            results = self.forward(main_images, context_images, return_intermediate=True)
            
            # Calcular probabilidades
            phase1_probs = F.softmax(results['phase1_logits'], dim=1)
            phase2_probs = F.softmax(results['phase2_logits'], dim=1)
            final_probs = F.softmax(results['final_logits'], dim=1)
            
            # Analizar diferencias
            phase1_to_phase2_change = torch.abs(phase2_probs - phase1_probs).mean(dim=1)
            phase2_to_final_change = torch.abs(final_probs - phase2_probs).mean(dim=1)
            
            return {
                'phase1_probs': phase1_probs,
                'phase2_probs': phase2_probs,
                'final_probs': final_probs,
                'phase1_to_phase2_change': phase1_to_phase2_change,
                'phase2_to_final_change': phase2_to_final_change,
                'confidence': results['confidence_score'],
                'attention_weights': results['attention_weights']
            }
    
    def freeze_previous_phases(self, freeze=True):
        """Congelar/descongelar fases anteriores"""
        for param in self.phase1_model.parameters():
            param.requires_grad = not freeze
        for param in self.phase2_model.parameters():
            param.requires_grad = not freeze
        
        self.freeze_previous_phases = freeze

class EnsembleCIFFNet(nn.Module):
    """Ensemble de múltiples modelos CIFF-Net para máxima robustez"""
    def __init__(self, model_configs, num_classes=7):
        super(EnsembleCIFFNet, self).__init__()
        
        self.models = nn.ModuleList()
        for config in model_configs:
            model = CIFFNetPhase3(**config)
            self.models.append(model)
        
        # Fusión del ensemble
        self.ensemble_fusion = nn.Sequential(
            nn.Linear(num_classes * len(model_configs), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
        
        # Pesos del ensemble
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(model_configs)) / len(model_configs)
        )
    
    def forward(self, main_images, context_images=None):
        """Forward pass del ensemble"""
        outputs = []
        confidences = []
        
        for model in self.models:
            output, confidence = model(main_images, context_images)
            outputs.append(output)
            confidences.append(confidence)
        
        # Ponderar salidas
        weighted_outputs = []
        normalized_weights = F.softmax(self.ensemble_weights, dim=0)
        
        for i, output in enumerate(outputs):
            weighted_outputs.append(output * normalized_weights[i])
        
        # Fusión final
        concatenated = torch.cat(outputs, dim=1)
        ensemble_output = self.ensemble_fusion(concatenated)
        
        # Confianza promedio
        avg_confidence = torch.stack(confidences).mean(dim=0)
        
        return ensemble_output, avg_confidence

def create_ciff_net_phase3(phase1_model_path, phase2_model_path, num_classes=7,
                          num_context_images=3, fusion_method='weighted_concatenation',
                          freeze_previous_phases=True):
    """Factory function para crear CIFF-Net Fase 3"""
    return CIFFNetPhase3(
        phase1_model_path, phase2_model_path, num_classes,
        num_context_images, fusion_method, freeze_previous_phases
    )

def create_ensemble_ciff_net(model_configs, num_classes=7):
    """Factory function para crear ensemble CIFF-Net"""
    return EnsembleCIFFNet(model_configs, num_classes)

def analyze_model_predictions(model, data_loader, device, class_names):
    """Analizar predicciones detalladas del modelo"""
    model.eval()
    
    phase_accuracies = {'phase1': [], 'phase2': [], 'final': []}
    confidence_scores = []
    prediction_changes = []
    
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                main_images = batch['main_images'].to(device)
                context_images = batch['context_images'].to(device)
                labels = batch['main_labels'].to(device)
            else:
                main_images, labels = batch
                main_images, labels = main_images.to(device), labels.to(device)
                context_images = None
            
            # Obtener análisis detallado
            analysis = model.get_phase_contributions(main_images, context_images)
            
            # Calcular accuracies por fase
            phase1_preds = analysis['phase1_probs'].argmax(dim=1)
            phase2_preds = analysis['phase2_probs'].argmax(dim=1)
            final_preds = analysis['final_probs'].argmax(dim=1)
            
            phase_accuracies['phase1'].extend((phase1_preds == labels).cpu().numpy())
            phase_accuracies['phase2'].extend((phase2_preds == labels).cpu().numpy())
            phase_accuracies['final'].extend((final_preds == labels).cpu().numpy())
            
            # Guardar métricas
            confidence_scores.extend(analysis['confidence'].cpu().numpy())
            prediction_changes.extend(analysis['phase2_to_final_change'].cpu().numpy())
    
    # Calcular estadísticas
    results = {
        'phase1_accuracy': np.mean(phase_accuracies['phase1']) * 100,
        'phase2_accuracy': np.mean(phase_accuracies['phase2']) * 100,
        'final_accuracy': np.mean(phase_accuracies['final']) * 100,
        'avg_confidence': np.mean(confidence_scores),
        'avg_prediction_change': np.mean(prediction_changes)
    }
    
    return results

if __name__ == "__main__":
    print("🧠 Probando CIFF-Net Fase 3 Completo...")
    
    try:
        # Crear modelo de Fase 3
        model = create_ciff_net_phase3(
            phase1_model_path='best_ciff_net_phase1.pth',
            phase2_model_path='best_ciff_net_phase2.pth',
            num_classes=7,
            fusion_method='weighted_concatenation'
        )
        
        # Test
        main_imgs = torch.randn(2, 3, 224, 224)
        context_imgs = torch.randn(2, 3, 3, 224, 224)
        
        # Forward pass
        final_output, confidence = model(main_imgs, context_imgs)
        print(f"✅ Final output shape: {final_output.shape}")
        print(f"✅ Confidence shape: {confidence.shape}")
        
        # Análisis detallado
        analysis = model.get_phase_contributions(main_imgs, context_imgs)
        print(f"📊 Análisis de contribuciones disponible")
        
        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Parámetros totales: {total_params:,}")
        print(f"📈 Parámetros entrenables: {trainable_params:,}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de entrenar Fase 1 y Fase 2 primero")