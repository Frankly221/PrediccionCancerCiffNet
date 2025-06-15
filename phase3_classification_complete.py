import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import math

class Phase3ClassifierComplete(nn.Module):
    """
    FASE 3 del paper CiffNet: Classification Network COMPLETA
    Optimizada para RTX 3070 con todas las técnicas del paper
    
    Implementa:
    1. Cliff-aware Classification
    2. Confidence Estimation  
    3. Multi-task Learning
    4. Adaptive Decision Making
    """
    
    def __init__(self, input_dim=256, num_classes=7, cliff_threshold=0.15):
        super(Phase3ClassifierComplete, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cliff_threshold = cliff_threshold
        
        print(f"🔧 FASE 3 COMPLETA - Classification según paper...")
        print(f"   Input dim: {input_dim} (desde Fase 2)")
        print(f"   Num classes: {num_classes}")
        print(f"   Cliff threshold: {cliff_threshold}")
        
        # ================================
        # 1. CLIFF-AWARE CLASSIFICATION (Paper)
        # ================================
        
        # Standard Classifier (para samples no-cliff)
        self.standard_classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Cliff-aware Classifier (para samples cliff)
        self.cliff_classifier = nn.Sequential(
            # Entrada: features + cliff_score + uncertainty info
            nn.Linear(input_dim + 1 + num_classes + 1, 512),  # +cliff +boundary +aleatoric
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
        
        # Feature Fusion for Classification (Paper method)
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, 512),  # enhanced + attended features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, input_dim),
            nn.ReLU(inplace=True)
        )
        
        # ================================
        # 2. CONFIDENCE ESTIMATION (Paper)
        # ================================
        
        # Predictive Confidence Estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim + num_classes + 1, 128),  # features + logits + cliff
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            
            nn.Linear(32, 1),
            nn.Sigmoid()  # Confidence [0,1]
        )
        
        # Epistemic Confidence (for uncertainty quantification)
        self.epistemic_confidence = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # High dropout for epistemic uncertainty
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
        # Aleatoric Confidence (for data uncertainty)
        self.aleatoric_confidence = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
            nn.Softplus()  # Always positive for variance
        )
        
        # ================================
        # 3. MULTI-TASK LEARNING (Paper)
        # ================================
        
        # Auxiliary Task 1: Cliff Detection Head
        self.cliff_detection_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary Task 2: Uncertainty Regression Head
        self.uncertainty_regression_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        
        # Auxiliary Task 3: Feature Quality Assessment
        self.quality_assessment_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Quality score [0,1]
        )
        
        # ================================
        # 4. ADAPTIVE DECISION MAKING (Paper)
        # ================================
        
        # Decision Fusion Network
        self.decision_fusion = nn.Sequential(
            nn.Linear(num_classes * 3 + 3, 256),  # 3 predictions + 3 confidences
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, num_classes)
        )
        
        # Adaptive Weight Generator (for fusion)
        self.adaptive_weights = nn.Sequential(
            nn.Linear(input_dim + 1, 64),  # features + cliff_score
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 3 weights for 3 classifiers
            nn.Softmax(dim=1)
        )
        
        # Final Prediction Calibrator
        self.prediction_calibrator = nn.Sequential(
            nn.Linear(num_classes + 1, 64),  # logits + confidence
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)
        )
        
        # ================================
        # NORMALIZATION & REGULARIZATION
        # ================================
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Batch normalization for logits
        self.logits_norm = nn.BatchNorm1d(num_classes)
        
        # Dropout layers for MC Dropout (epistemic uncertainty)
        self.mc_dropout1 = nn.Dropout(0.5)
        self.mc_dropout2 = nn.Dropout(0.3)
        
        self._initialize_weights()
        
        print(f"✅ FASE 3 COMPLETA inicializada:")
        print(f"   🎯 Cliff-aware Classification: Standard + Cliff classifiers")
        print(f"   📊 Confidence Estimation: Predictive + Epistemic + Aleatoric")  
        print(f"   🔄 Multi-task Learning: 3 auxiliary tasks")
        print(f"   🧠 Adaptive Decision: Fusion + Calibration")
        print(f"   🔧 Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """Inicialización específica del paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform para clasificadores
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def monte_carlo_dropout_prediction(self, features, n_samples=10):
        """
        Monte Carlo Dropout para epistemic uncertainty (Paper method)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            # Forward pass with dropout
            x = self.mc_dropout1(features)
            x = self.mc_dropout2(x)
            pred = self.epistemic_confidence(x)
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [n_samples, B, num_classes]
        
        # Mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        self.eval()  # Disable dropout
        return mean_pred, epistemic_uncertainty
    
    def compute_predictive_entropy(self, logits):
        """
        Compute predictive entropy (Paper uncertainty measure)
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # Predictive entropy
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        
        # Normalize entropy [0,1]
        max_entropy = math.log(self.num_classes)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    def adaptive_classification(self, features, cliff_info):
        """
        Adaptive classification based on cliff detection (Paper core)
        """
        cliff_score = cliff_info['cliff_score']
        cliff_mask = cliff_info['cliff_mask']
        boundary_distances = cliff_info['boundary_distances']
        aleatoric_uncertainty = cliff_info['aleatoric_uncertainty']
        
        batch_size = features.size(0)
        
        # ================================
        # 1. MULTIPLE CLASSIFIER PREDICTIONS
        # ================================
        
        # Standard classifier (always)
        standard_logits = self.standard_classifier(features)
        
        # Cliff-aware classifier input
        cliff_input = torch.cat([
            features,                    # [B, 256]
            cliff_score,                 # [B, 1]
            boundary_distances,          # [B, num_classes]
            aleatoric_uncertainty        # [B, 1]
        ], dim=1)
        
        cliff_logits = self.cliff_classifier(cliff_input)
        
        # Epistemic classifier (with MC Dropout)
        epistemic_logits, epistemic_var = self.monte_carlo_dropout_prediction(features)
        
        # ================================
        # 2. CONFIDENCE ESTIMATION
        # ================================
        
        # Standard confidence
        standard_entropy = self.compute_predictive_entropy(standard_logits)
        standard_confidence = 1.0 - standard_entropy
        
        # Cliff confidence  
        cliff_entropy = self.compute_predictive_entropy(cliff_logits)
        cliff_confidence = 1.0 - cliff_entropy
        
        # Epistemic confidence
        epistemic_entropy = self.compute_predictive_entropy(epistemic_logits)
        epistemic_confidence = 1.0 - epistemic_entropy
        
        # Predictive confidence (learned)
        conf_input = torch.cat([features, standard_logits, cliff_score], dim=1)
        predictive_confidence = self.confidence_estimator(conf_input)
        
        # ================================
        # 3. ADAPTIVE FUSION WEIGHTS
        # ================================
        
        # Generate adaptive weights based on cliff info
        weight_input = torch.cat([features, cliff_score], dim=1)
        fusion_weights = self.adaptive_weights(weight_input)  # [B, 3]
        
        # Separate weights
        w_standard = fusion_weights[:, 0:1]      # [B, 1]
        w_cliff = fusion_weights[:, 1:2]         # [B, 1]
        w_epistemic = fusion_weights[:, 2:3]     # [B, 1]
        
        # ================================
        # 4. DECISION FUSION
        # ================================
        
        # Weighted logits combination
        fused_logits = (w_standard * standard_logits + 
                       w_cliff * cliff_logits +
                       w_epistemic * epistemic_logits)
        
        # Alternative: Decision fusion network
        fusion_input = torch.cat([
            standard_logits,        # [B, num_classes]
            cliff_logits,          # [B, num_classes]  
            epistemic_logits,      # [B, num_classes]
            standard_confidence,   # [B, 1]
            cliff_confidence,      # [B, 1]
            epistemic_confidence   # [B, 1]
        ], dim=1)
        
        fusion_logits = self.decision_fusion(fusion_input)
        
        # ================================
        # 5. FINAL PREDICTION CALIBRATION
        # ================================
        
        # Calibrate final prediction
        final_confidence = (w_standard * standard_confidence +
                          w_cliff * cliff_confidence +
                          w_epistemic * epistemic_confidence)
        
        calib_input = torch.cat([fusion_logits, final_confidence], dim=1)
        calibrated_logits = self.prediction_calibrator(calib_input)
        
        return {
            'standard_logits': standard_logits,
            'cliff_logits': cliff_logits,
            'epistemic_logits': epistemic_logits,
            'fused_logits': fused_logits,
            'fusion_logits': fusion_logits,
            'calibrated_logits': calibrated_logits,
            'fusion_weights': fusion_weights,
            'confidences': {
                'standard': standard_confidence,
                'cliff': cliff_confidence,
                'epistemic': epistemic_confidence,
                'predictive': predictive_confidence,
                'final': final_confidence
            },
            'epistemic_variance': epistemic_var
        }
    
    def forward(self, phase2_outputs, return_all=False):
        """
        Forward pass COMPLETO según paper CiffNet Fase 3
        
        Args:
            phase2_outputs: dict con outputs de Fase 2
            return_all: bool, si retornar análisis completo
        """
        # Extract Phase 2 outputs
        enhanced_features = phase2_outputs['enhanced_features']
        cliff_score = phase2_outputs['cliff_score']
        cliff_mask = phase2_outputs['cliff_mask']
        
        # Prepare cliff info
        cliff_info = {
            'cliff_score': cliff_score,
            'cliff_mask': cliff_mask,
            'boundary_distances': phase2_outputs['boundary_distances'],
            'aleatoric_uncertainty': phase2_outputs['aleatoric_uncertainty']
        }
        
        # Normalize features
        features = self.layer_norm(enhanced_features)
        
        with autocast():
            # ================================
            # ADAPTIVE CLASSIFICATION
            # ================================
            
            classification_results = self.adaptive_classification(features, cliff_info)
            
            # ================================
            # MULTI-TASK PREDICTIONS
            # ================================
            
            # Auxiliary task 1: Cliff detection
            aux_cliff_pred = self.cliff_detection_head(features)
            
            # Auxiliary task 2: Uncertainty regression
            aux_uncertainty_pred = self.uncertainty_regression_head(features)
            
            # Auxiliary task 3: Feature quality
            aux_quality_pred = self.quality_assessment_head(features)
            
            # ================================
            # FINAL OUTPUTS
            # ================================
            
            # Main prediction (calibrated)
            final_logits = classification_results['calibrated_logits']
            
            # Apply batch normalization
            final_logits = self.logits_norm(final_logits)
            
            # Final probabilities
            final_probs = F.softmax(final_logits, dim=1)
            
            # Final prediction
            final_prediction = torch.argmax(final_logits, dim=1)
            
            # Overall confidence
            final_confidence = classification_results['confidences']['final']
            
            # Prediction entropy
            prediction_entropy = self.compute_predictive_entropy(final_logits)
        
        # ================================
        # ANALYSIS & METRICS
        # ================================
        
        batch_size = features.size(0)
        cliff_samples = int(cliff_mask.sum().item())
        
        analysis = {
            'batch_size': batch_size,
            'cliff_samples': cliff_samples,
            'cliff_ratio': float(cliff_samples / batch_size),
            'avg_confidence': float(final_confidence.mean().item()),
            'avg_entropy': float(prediction_entropy.mean().item()),
            'prediction_distribution': {
                int(i): int((final_prediction == i).sum().item()) 
                for i in range(self.num_classes)
            },
            'confidence_stats': {
                'mean': float(final_confidence.mean().item()),
                'std': float(final_confidence.std().item()),
                'min': float(final_confidence.min().item()),
                'max': float(final_confidence.max().item())
            },
            'fusion_weights_avg': {
                'standard': float(classification_results['fusion_weights'][:, 0].mean().item()),
                'cliff': float(classification_results['fusion_weights'][:, 1].mean().item()),
                'epistemic': float(classification_results['fusion_weights'][:, 2].mean().item())
            }
        }
        
        # Base return
        result = {
            'logits': final_logits,                    # [B, num_classes] - Para loss
            'probabilities': final_probs,              # [B, num_classes] - Probabilidades
            'predictions': final_prediction,           # [B] - Predicciones finales
            'confidence': final_confidence,            # [B, 1] - Confianza final
            'prediction_entropy': prediction_entropy,  # [B, 1] - Entropía predictiva
            
            # Multi-task outputs
            'aux_cliff_pred': aux_cliff_pred,          # [B, 1] - Predicción cliff auxiliar
            'aux_uncertainty_pred': aux_uncertainty_pred, # [B, 1] - Predicción uncertainty
            'aux_quality_pred': aux_quality_pred,      # [B, 1] - Calidad features
            
            # Analysis
            'analysis': analysis
        }
        
        # Extended return for training/analysis
        if return_all:
            result.update({
                'classification_breakdown': classification_results,
                'all_confidences': classification_results['confidences'],
                'epistemic_variance': classification_results['epistemic_variance'],
                'fusion_weights': classification_results['fusion_weights']
            })
        
        return result

def create_phase3_complete_classifier(input_dim=256, num_classes=7, cliff_threshold=0.15):
    """
    Factory para crear Fase 3 COMPLETA según paper
    """
    classifier = Phase3ClassifierComplete(
        input_dim=input_dim,
        num_classes=num_classes,
        cliff_threshold=cliff_threshold
    )
    
    return classifier

# LOSS FUNCTION COMPLETA DEL PAPER
class CiffNetPhase3Loss(nn.Module):
    """
    Loss function COMPLETA para Fase 3 según paper CiffNet
    Multi-task loss con pesos adaptativos
    """
    
    def __init__(self, num_classes=7, alpha=1.0, beta=0.3, gamma=0.2, delta=0.15, epsilon=0.1):
        super(CiffNetPhase3Loss, self).__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha      # Main classification loss
        self.beta = beta        # Confidence loss  
        self.gamma = gamma      # Auxiliary cliff loss
        self.delta = delta      # Auxiliary uncertainty loss
        self.epsilon = epsilon  # Auxiliary quality loss
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()
        self.auxiliary_loss = nn.BCELoss()
        self.uncertainty_loss = nn.MSELoss()
        self.quality_loss = nn.MSELoss()
        
        print(f"✅ CiffNet Phase 3 Loss inicializado:")
        print(f"   α (classification): {alpha}")
        print(f"   β (confidence): {beta}")
        print(f"   γ (aux cliff): {gamma}")
        print(f"   δ (aux uncertainty): {delta}")
        print(f"   ε (aux quality): {epsilon}")
    
    def compute_confidence_target(self, logits, targets):
        """
        Compute target confidence based on prediction correctness
        """
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            correct_mask = (predictions == targets).float()
            
            # High confidence for correct, low for incorrect
            confidence_target = correct_mask * 0.9 + (1 - correct_mask) * 0.1
            
        return confidence_target.unsqueeze(1)
    
    def compute_uncertainty_target(self, logits, targets):
        """
        Compute target uncertainty based on prediction entropy
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Normalize entropy as uncertainty target
            max_entropy = math.log(self.num_classes)
            uncertainty_target = entropy / max_entropy
            
        return uncertainty_target.unsqueeze(1)
    
    def compute_quality_target(self, confidence, cliff_score):
        """
        Compute feature quality target
        """
        with torch.no_grad():
            # High quality = high confidence + low cliff score
            quality_target = confidence * (1.0 - cliff_score)
            
        return quality_target
    
    def forward(self, phase3_outputs, targets, cliff_targets=None):
        """
        Compute total Phase 3 loss
        """
        # Extract outputs
        logits = phase3_outputs['logits']
        confidence = phase3_outputs['confidence']
        aux_cliff_pred = phase3_outputs['aux_cliff_pred']
        aux_uncertainty_pred = phase3_outputs['aux_uncertainty_pred']
        aux_quality_pred = phase3_outputs['aux_quality_pred']
        
        # Main classification loss
        main_loss = self.classification_loss(logits, targets)
        
        # Confidence loss
        confidence_target = self.compute_confidence_target(logits, targets)
        conf_loss = self.confidence_loss(confidence, confidence_target)
        
        # Auxiliary losses
        if cliff_targets is None:
            # Use prediction difficulty as cliff target
            cliff_target = self.compute_uncertainty_target(logits, targets)
        else:
            cliff_target = cliff_targets
        
        aux_cliff_loss = self.auxiliary_loss(aux_cliff_pred, cliff_target)
        
        # Uncertainty target
        uncertainty_target = self.compute_uncertainty_target(logits, targets)
        aux_uncertainty_loss = self.uncertainty_loss(aux_uncertainty_pred, uncertainty_target)
        
        # Quality target
        quality_target = self.compute_quality_target(confidence, aux_cliff_pred)
        aux_quality_loss = self.quality_loss(aux_quality_pred, quality_target)
        
        # Total loss
        total_loss = (self.alpha * main_loss +
                     self.beta * conf_loss +
                     self.gamma * aux_cliff_loss +
                     self.delta * aux_uncertainty_loss +
                     self.epsilon * aux_quality_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': main_loss,
            'confidence_loss': conf_loss,
            'aux_cliff_loss': aux_cliff_loss,
            'aux_uncertainty_loss': aux_uncertainty_loss,
            'aux_quality_loss': aux_quality_loss
        }

if __name__ == "__main__":
    # Test implementación completa Fase 3
    print("🔬 TESTING FASE 3 COMPLETA SEGÚN PAPER")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear clasificador completo
    classifier = create_phase3_complete_classifier(
        input_dim=256,
        num_classes=7,
        cliff_threshold=0.15
    ).to(device)
    
    # Simular outputs de Fase 2
    batch_size = 4
    phase2_outputs = {
        'enhanced_features': torch.randn(batch_size, 256).to(device),
        'cliff_score': torch.rand(batch_size, 1).to(device),
        'cliff_mask': torch.randint(0, 2, (batch_size, 1)).bool().to(device),
        'boundary_distances': torch.rand(batch_size, 7).to(device),
        'aleatoric_uncertainty': torch.rand(batch_size, 1).to(device)
    }
    
    test_labels = torch.randint(0, 7, (batch_size,)).to(device)
    
    # Forward pass
    classifier.eval()
    with torch.no_grad():
        results = classifier(phase2_outputs, return_all=True)
    
    print(f"\n✅ FASE 3 COMPLETA funcionando:")
    print(f"   Logits: {results['logits'].shape}")
    print(f"   Predictions: {results['predictions'].shape}")
    print(f"   Confidence: {results['confidence'].shape}")
    print(f"   Analysis: {len(results['analysis'])} métricas")
    
    # Test loss
    loss_fn = CiffNetPhase3Loss(num_classes=7)
    loss_results = loss_fn(results, test_labels)
    
    print(f"   Loss total: {loss_results['total_loss'].item():.4f}")
    print(f"\n🎯 FASE 3 PAPER COMPLETA ✅")