import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import logging
logger = logging.getLogger(__name__)

class CliffAwareClassifier(nn.Module):
    """
    Clasificador que toma decisiones adaptativas basadas en cliff detection
    """
    
    def __init__(self, input_dim, num_classes, cliff_threshold=0.15):
        super(CliffAwareClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cliff_threshold = cliff_threshold
        
        # Capas de clasificación principales
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Clasificador especializado para casos cliff
        self.cliff_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
        
        # Uncertainty estimation con Monte Carlo Dropout
        self.uncertainty_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Higher dropout for uncertainty
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        print(f"✅ CliffAwareClassifier creado:")
        print(f"   Input dim: {input_dim}")
        print(f"   Num classes: {num_classes}")
        print(f"   Cliff threshold: {cliff_threshold}")
    
    def forward(self, enhanced_features, cliff_scores, training=True):
        batch_size = enhanced_features.size(0)
        
        # ✅ CORREGIDO: Asegurar dimensiones correctas de cliff_scores
        if cliff_scores.dim() > 1:
            cliff_scores = cliff_scores.squeeze(-1)  # [batch_size, 1] → [batch_size]
        
        # Clasificación principal
        main_logits = self.classifier(enhanced_features)
        
        # Clasificación especializada para cliff
        cliff_logits = self.cliff_classifier(enhanced_features)
        
        # ✅ DEBUGGING: Verificar que los logits tengan sentido
        logger.info(f"🔍 Main logits shape: {main_logits.shape}")
        logger.info(f"🔍 Main logits values: {main_logits}")
        logger.info(f"🔍 Cliff logits shape: {cliff_logits.shape}")
        logger.info(f"🔍 Cliff logits values: {cliff_logits}")
        
        # Uncertainty estimation
        uncertainty_scores = self.uncertainty_layers(enhanced_features).squeeze(-1)
        
        # cliff_mask ahora será [batch_size]
        cliff_mask = cliff_scores > self.cliff_threshold
        
        # Combinar logits basado en cliff detection
        final_logits = main_logits.clone()
        if cliff_mask.any():
            final_logits[cliff_mask] = cliff_logits[cliff_mask]
        
        # ✅ DEBUGGING: Verificar final_logits
        logger.info(f"🔍 Final logits shape: {final_logits.shape}")
        logger.info(f"🔍 Final logits values: {final_logits}")
        
        # Aplicar softmax para probabilidades
        probabilities = F.softmax(final_logits, dim=1)
        
        # ✅ DEBUGGING: Verificar probabilidades
        logger.info(f"🔍 Probabilities shape: {probabilities.shape}")
        logger.info(f"🔍 Probabilities values: {probabilities}")
        logger.info(f"🔍 Probabilities sum: {probabilities.sum(dim=1)}")
        
        # Monte Carlo Dropout para uncertainty (solo en training)
        if training and self.training:
            mc_predictions = []
            n_samples = 10
            
            for _ in range(n_samples):
                mc_logits = self.classifier(enhanced_features)
                mc_probs = F.softmax(mc_logits, dim=1)
                mc_predictions.append(mc_probs.unsqueeze(0))
            
            mc_predictions = torch.cat(mc_predictions, dim=0)  # [n_samples, batch_size, num_classes]
            
            # Predictive entropy como medida de incertidumbre
            mean_probs = mc_predictions.mean(dim=0)
            predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        else:
            predictive_entropy = uncertainty_scores
        
        # Confidence como max probability
        confidence_scores = torch.max(probabilities, dim=1)[0]
        
        # ✅ CORREGIDO: Predictions finales CORRECTAS
        predictions = torch.argmax(final_logits, dim=1)
        
        # ✅ DEBUGGING: Verificar predictions
        logger.info(f"🔍 Predictions shape: {predictions.shape}")
        logger.info(f"🔍 Predictions values: {predictions}")
        logger.info(f"🔍 Confidence shape: {confidence_scores.shape}")
        logger.info(f"🔍 Confidence values: {confidence_scores}")
        
        return {
            'logits': final_logits,
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence': confidence_scores,
            'uncertainty': predictive_entropy,
            'main_logits': main_logits,
            'cliff_logits': cliff_logits,
            'cliff_mask': cliff_mask
        }

class CiffNetPhase3Loss(nn.Module):
    """
    Loss function para Phase 3 - CORREGIDO para autocast
    """
    
    def __init__(self, num_classes, alpha=1.0, beta=0.3, gamma=0.2):
        super(CiffNetPhase3Loss, self).__init__()
        
        self.num_classes = num_classes
        self.alpha = alpha  # Peso para classification loss
        self.beta = beta    # Peso para uncertainty loss
        self.gamma = gamma  # Peso para confidence loss
        
        # ✅ CORREGIDO: Usar CrossEntropyLoss en lugar de BCELoss
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Loss para uncertainty (regression)
        self.uncertainty_loss = nn.MSELoss()
        
        print(f"✅ CiffNetPhase3Loss inicializado:")
        print(f"   Alpha (classification): {alpha}")
        print(f"   Beta (uncertainty): {beta}")
        print(f"   Gamma (confidence): {gamma}")
    
    def forward(self, outputs, targets):
        """
        Compute loss combinado para Phase 3
        """
        # Extract outputs
        logits = outputs['logits']
        probabilities = outputs['probabilities']
        predictions = outputs['predictions']
        confidence = outputs['confidence']
        uncertainty = outputs['uncertainty']
        main_logits = outputs['main_logits']
        cliff_logits = outputs['cliff_logits']
        cliff_mask = outputs['cliff_mask']
        
        batch_size = targets.size(0)
        device = targets.device
        
        # ================================
        # 1. CLASSIFICATION LOSS (Principal)
        # ================================
        # ✅ CORREGIDO: CrossEntropyLoss es safe para autocast
        classification_loss = self.classification_loss(logits, targets)
        
        # Loss adicional para main classifier
        main_loss = self.classification_loss(main_logits, targets)
        
        # Loss para cliff classifier (solo en casos cliff)
        if cliff_mask.any():
            cliff_targets = targets[cliff_mask]
            cliff_loss = self.classification_loss(cliff_logits[cliff_mask], cliff_targets)
        else:
            cliff_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # ================================
        # 2. UNCERTAINTY LOSS
        # ================================
        # Target uncertainty: alta para predicciones incorrectas
        correct_predictions = (predictions == targets).float()
        target_uncertainty = 1.0 - correct_predictions  # [0,1]: 1 = uncertain, 0 = certain
        
        uncertainty_loss_val = self.uncertainty_loss(uncertainty, target_uncertainty)
        
        # ================================
        # 3. CONFIDENCE CALIBRATION LOSS
        # ================================
        # Queremos que confidence correlacione con accuracy
        confidence_loss_val = self.uncertainty_loss(confidence, correct_predictions)
        
        # ================================
        # 4. CONSISTENCY LOSS
        # ================================
        # Consistency entre main y cliff classifiers
        main_probs = F.softmax(main_logits, dim=1)
        cliff_probs = F.softmax(cliff_logits, dim=1)
        
        # KL divergence para consistency
        kl_loss = F.kl_div(
            F.log_softmax(cliff_logits, dim=1),
            main_probs,
            reduction='batchmean'
        )
        
        # ================================
        # 5. COMBINE ALL LOSSES
        # ================================
        total_loss = (
            self.alpha * classification_loss +
            0.5 * main_loss +
            0.3 * cliff_loss +
            self.beta * uncertainty_loss_val +
            self.gamma * confidence_loss_val +
            0.1 * kl_loss
        )
        
        # Loss breakdown para debugging
        loss_breakdown = {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'main_loss': main_loss,
            'cliff_loss': cliff_loss,
            'uncertainty_loss': uncertainty_loss_val,
            'confidence_loss': confidence_loss_val,
            'consistency_loss': kl_loss
        }
        
        return loss_breakdown

class CiffNetPhase3CompleteClassifier(nn.Module):
    """
    Phase 3 completa con todas las características del paper
    """
    
    def __init__(self, input_dim, num_classes, cliff_threshold=0.15):
        super(CiffNetPhase3CompleteClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cliff_threshold = cliff_threshold
        
        # Cliff-aware classifier
        self.cliff_aware_classifier = CliffAwareClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            cliff_threshold=cliff_threshold
        )
        
        # Feature refinement antes de clasificación
        self.feature_refiner = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        
        print(f"✅ CiffNetPhase3CompleteClassifier creado:")
        print(f"   Input dim: {input_dim}")
        print(f"   Num classes: {num_classes}")
        print(f"   Cliff threshold: {cliff_threshold}")
    
    def forward(self, phase2_outputs, return_all=False):
        """
        Forward pass de Phase 3
        
        Args:
            phase2_outputs: Outputs de Phase 2 (dict)
            return_all: Si devolver información detallada
        """
        # Extract Phase 2 outputs
        enhanced_features = phase2_outputs['enhanced_features']
        cliff_scores = phase2_outputs['cliff_score']
        
        batch_size = enhanced_features.size(0)
        
        # Feature refinement
        refined_features = self.feature_refiner(enhanced_features)
        
        # Cliff-aware classification
        classification_outputs = self.cliff_aware_classifier(
            refined_features, 
            cliff_scores, 
            training=self.training
        )
        
        # Resultado principal
        result = {
            'logits': classification_outputs['logits'],
            'probabilities': classification_outputs['probabilities'],
            'predictions': classification_outputs['predictions'],
            'confidence': classification_outputs['confidence'],
            'uncertainty': classification_outputs['uncertainty']
        }
        
        # Información detallada si se requiere
        if return_all:
            result.update({
                'enhanced_features': enhanced_features,
                'refined_features': refined_features,
                'cliff_scores': cliff_scores,
                'main_logits': classification_outputs['main_logits'],
                'cliff_logits': classification_outputs['cliff_logits'],
                'cliff_mask': classification_outputs['cliff_mask']
            })
        
        return result

def create_phase3_complete_classifier(input_dim=256, num_classes=7, cliff_threshold=0.15):
    """
    Factory function para crear Phase 3 completa
    """
    print("🔧 Creando Phase 3 Complete Classifier...")
    
    model = CiffNetPhase3CompleteClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        cliff_threshold=cliff_threshold
    )
    
    # Inicialización de pesos
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    print(f"✅ Phase 3 Complete creada:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parámetros totales: {total_params:,}")
    
    return model

# Test function
def test_phase3_complete():
    """
    Test básico para verificar funcionamiento
    """
    print("🧪 Testing Phase 3 Complete...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo
    model = create_phase3_complete_classifier()
    model = model.to(device)
    
    # Datos de prueba (simulando outputs de Phase 2)
    batch_size = 16
    input_dim = 256
    num_classes = 7
    
    # Simular phase2_outputs
    phase2_outputs = {
        'enhanced_features': torch.randn(batch_size, input_dim).to(device),
        'cliff_score': torch.rand(batch_size).to(device)  # ✅ CORREGIDO: [batch_size] no [batch_size, 1]
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(phase2_outputs, return_all=True)
    
    print(f"✅ Test exitoso:")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Predictions shape: {outputs['predictions'].shape}")
    print(f"   Confidence range: {outputs['confidence'].min():.3f} - {outputs['confidence'].max():.3f}")
    
    # Test loss
    loss_fn = CiffNetPhase3Loss(num_classes=num_classes)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    model.train()
    outputs_train = model(phase2_outputs, return_all=True)
    loss_dict = loss_fn(outputs_train, targets)
    
    print(f"✅ Loss test exitoso:")
    print(f"   Total loss: {loss_dict['total_loss']:.4f}")
    print(f"   Classification loss: {loss_dict['classification_loss']:.4f}")
    
    return True

if __name__ == "__main__":
    # Ejecutar test
    test_phase3_complete()
    print("🎯 Phase 3 Complete funcionando correctamente!")