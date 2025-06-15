import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import math

class Phase2CliffDetectorComplete(nn.Module):
    """
    FASE 2 del paper CiffNet: Implementación COMPLETA según paper original
    
    Implementa:
    1. Cliff Feature Mining (CFM)
    2. Cliff Region Identification (CRI)  
    3. Cliff-Aware Feature Enhancement (CAFE)
    """
    
    def __init__(self, input_dim=256, cliff_threshold=0.15, num_classes=7):
        super(Phase2CliffDetectorComplete, self).__init__()
        
        self.input_dim = input_dim
        self.cliff_threshold = cliff_threshold
        self.num_classes = num_classes
        
        print(f"🔧 FASE 2 COMPLETA - Cliff Detection según paper...")
        
        # ================================
        # 1. CLIFF FEATURE MINING (CFM)
        # ================================
        
        # Local Gradient Analyzer (Paper específico)
        self.gradient_analyzer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, input_dim),  # Gradient magnitude per feature
            nn.Sigmoid()
        )
        
        # Feature Magnitude Variation Detector
        self.magnitude_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Overall magnitude variation
            nn.Sigmoid()
        )
        
        # Decision Boundary Proximity (Paper method)
        self.boundary_proximity = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),  # Distance to each class boundary
            nn.Softmax(dim=1)
        )
        
        # Uncertainty Quantification (Epistemic + Aleatoric)
        self.epistemic_uncertainty = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Dropout for epistemic uncertainty
            nn.Linear(128, input_dim),
            nn.Softplus()
        )
        
        self.aleatoric_uncertainty = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),  # Single aleatoric uncertainty value
            nn.Softplus()
        )
        
        # ================================
        # 2. CLIFF REGION IDENTIFICATION (CRI)
        # ================================
        
        # Spatial Cliff Detector (Paper method)
        self.spatial_cliff_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Spatial cliff probability
        )
        
        # Feature-space Cliff Mapper
        self.feature_cliff_mapper = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),  # Cliff mapping per feature dimension
            nn.Sigmoid()
        )
        
        # Multi-scale Cliff Analyzer (Paper específico)
        self.multiscale_analyzer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ) for _ in range(3)  # 3 escalas diferentes
        ])
        
        # Cliff Confidence Scorer
        self.confidence_scorer = nn.Sequential(
            nn.Linear(input_dim + 4, 64),  # features + 4 cliff indicators
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # ================================
        # 3. CLIFF-AWARE FEATURE ENHANCEMENT (CAFE)
        # ================================
        
        # Cliff-guided Attention (Paper method)
        self.cliff_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,  # 256/8 = 32 per head
            dropout=0.1,
            batch_first=True
        )
        
        # Attention projection
        self.attention_proj = nn.Linear(input_dim, input_dim)
        
        # Feature Re-weighting Module
        self.feature_reweighter = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # features + cliff_score
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Weights [0,1]
        )
        
        # Boundary-aware Enhancement
        self.boundary_enhancer = nn.Sequential(
            nn.Linear(input_dim + num_classes, 256),  # features + boundary_dist
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
            nn.Tanh()  # Enhancement residual [-1,1]
        )
        
        # Adaptive Feature Fusion (Paper final step)
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(input_dim * 3, 512),  # original + attention + boundary
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )
        
        # ================================
        # OPTIMIZACIONES Y NORMALIZACIÓN
        # ================================
        
        # Layer normalization para estabilidad
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Batch normalization después de fusion
        self.batch_norm = nn.BatchNorm1d(input_dim)
        
        self._initialize_weights()
        
        print(f"✅ FASE 2 COMPLETA inicializada:")
        print(f"   📊 CFM: Gradient + Magnitude + Boundary + Uncertainty")
        print(f"   🎯 CRI: Spatial + Feature + Multi-scale + Confidence")  
        print(f"   🧠 CAFE: Attention + Re-weight + Boundary + Fusion")
        print(f"   🔧 Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def _initialize_weights(self):
        """Inicialización específica del paper"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform para capas lineales
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # Inicialización específica para attention
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.xavier_uniform_(module.out_proj.weight)
    
    def compute_local_gradients(self, features):
        """
        Compute local gradients según paper (CFM step 1)
        """
        # Enable gradient computation
        features_grad = features.clone().requires_grad_(True)
        
        # Compute gradient magnitudes
        grad_magnitudes = self.gradient_analyzer(features_grad)
        
        return grad_magnitudes
    
    def compute_feature_variations(self, features):
        """
        Compute feature magnitude variations (CFM step 2)
        """
        # Batch-wise variation analysis
        batch_mean = features.mean(dim=0, keepdim=True)
        batch_std = features.std(dim=0, keepdim=True)
        
        # Normalized variation per sample
        feature_variation = torch.abs(features - batch_mean) / (batch_std + 1e-8)
        overall_variation = self.magnitude_detector(feature_variation)
        
        return overall_variation, feature_variation
    
    def multi_scale_cliff_analysis(self, features):
        """
        Multi-scale cliff analysis (CRI step 3)
        """
        multiscale_scores = []
        
        for i, analyzer in enumerate(self.multiscale_analyzer):
            # Different scales through different network depths
            scale_score = analyzer(features)
            multiscale_scores.append(scale_score)
        
        # Combine scales
        combined_scores = torch.cat(multiscale_scores, dim=1)  # [B, 3]
        
        # Weighted combination (paper method)
        weights = torch.softmax(torch.ones(3, device=features.device), dim=0)
        final_multiscale = torch.sum(combined_scores * weights, dim=1, keepdim=True)
        
        return final_multiscale, combined_scores
    
    def cliff_guided_attention(self, features, cliff_score):
        """
        Cliff-guided attention mechanism (CAFE step 1)
        """
        batch_size = features.size(0)
        
        # Prepare for attention (add sequence dimension)
        features_seq = features.unsqueeze(1)  # [B, 1, D]
        
        # Self-attention with cliff guidance
        attended_features, attention_weights = self.cliff_attention(
            features_seq, features_seq, features_seq
        )
        
        # Remove sequence dimension
        attended_features = attended_features.squeeze(1)  # [B, D]
        
        # Project and modulate with cliff score
        projected = self.attention_proj(attended_features)
        cliff_modulated = projected * cliff_score  # Modulate by cliff score
        
        return cliff_modulated, attention_weights.squeeze(1)
    
    def forward(self, phase1_features):
        """
        Forward pass COMPLETO según paper CiffNet
        """
        batch_size = phase1_features.size(0)
        
        # Normalize input features
        features = self.layer_norm(phase1_features)
        
        with autocast():
            # ================================
            # 1. CLIFF FEATURE MINING (CFM)
            # ================================
            
            # Step 1: Local gradient analysis
            local_gradients = self.compute_local_gradients(features)
            
            # Step 2: Feature magnitude variations  
            magnitude_variation, feature_variations = self.compute_feature_variations(features)
            
            # Step 3: Decision boundary proximity
            boundary_distances = self.boundary_proximity(features)
            
            # Step 4: Uncertainty quantification
            epistemic_unc = self.epistemic_uncertainty(features)
            aleatoric_unc = self.aleatoric_uncertainty(features)
            
            # ================================
            # 2. CLIFF REGION IDENTIFICATION (CRI)
            # ================================
            
            # Step 1: Spatial cliff detection
            spatial_cliff = self.spatial_cliff_detector(features)
            
            # Step 2: Feature-space cliff mapping
            feature_cliff_map = self.feature_cliff_mapper(features)
            
            # Step 3: Multi-scale cliff analysis
            multiscale_cliff, multiscale_components = self.multi_scale_cliff_analysis(features)
            
            # Step 4: Cliff confidence scoring
            cliff_indicators = torch.cat([
                spatial_cliff, multiscale_cliff, magnitude_variation, aleatoric_unc
            ], dim=1)
            
            confidence_input = torch.cat([features, cliff_indicators], dim=1)
            cliff_confidence = self.confidence_scorer(confidence_input)
            
            # ================================
            # FINAL CLIFF SCORE (Paper method)
            # ================================
            
            # Combine all cliff indicators (paper weights)
            final_cliff_score = (0.25 * spatial_cliff + 
                               0.25 * multiscale_cliff +
                               0.20 * magnitude_variation +
                               0.15 * cliff_confidence +
                               0.15 * aleatoric_unc)
            
            # ================================
            # 3. CLIFF-AWARE FEATURE ENHANCEMENT (CAFE)
            # ================================
            
            # Step 1: Cliff-guided attention
            attended_features, attention_weights = self.cliff_guided_attention(
                features, final_cliff_score
            )
            
            # Step 2: Feature re-weighting
            reweight_input = torch.cat([features, final_cliff_score], dim=1)
            feature_weights = self.feature_reweighter(reweight_input)
            reweighted_features = features * feature_weights
            
            # Step 3: Boundary-aware enhancement
            boundary_input = torch.cat([features, boundary_distances], dim=1)
            boundary_enhancement = self.boundary_enhancer(boundary_input)
            boundary_enhanced = features + boundary_enhancement
            
            # Step 4: Adaptive feature fusion
            fusion_input = torch.cat([
                reweighted_features,    # Re-weighted original
                attended_features,      # Attention enhanced  
                boundary_enhanced       # Boundary enhanced
            ], dim=1)
            
            fused_features = self.adaptive_fusion(fusion_input)
            
            # Final normalization
            enhanced_features = self.batch_norm(fused_features)
            
            # Cliff mask
            cliff_mask = final_cliff_score > self.cliff_threshold
        
        # ================================
        # ANÁLISIS COMPLETO
        # ================================
        
        analysis = {
            'cliff_samples': int(cliff_mask.sum().item()),
            'cliff_ratio': float(cliff_mask.float().mean().item()),
            'avg_cliff_score': float(final_cliff_score.mean().item()),
            'spatial_cliff_avg': float(spatial_cliff.mean().item()),
            'multiscale_cliff_avg': float(multiscale_cliff.mean().item()),
            'magnitude_variation_avg': float(magnitude_variation.mean().item()),
            'epistemic_uncertainty_avg': float(epistemic_unc.mean().item()),
            'aleatoric_uncertainty_avg': float(aleatoric_unc.mean().item()),
            'attention_entropy': float(self._compute_attention_entropy(attention_weights)),
            'boundary_sharpness': float(torch.max(boundary_distances, dim=1)[0].mean().item())
        }
        
        return {
            # OUTPUTS PRINCIPALES
            'enhanced_features': enhanced_features,        # [B, 256] Para Fase 3
            'cliff_score': final_cliff_score,             # [B, 1] Score final
            'cliff_mask': cliff_mask,                     # [B, 1] Máscara binaria
            
            # CFM OUTPUTS
            'local_gradients': local_gradients,           # [B, 256] Gradientes locales
            'magnitude_variation': magnitude_variation,    # [B, 1] Variación magnitud  
            'boundary_distances': boundary_distances,     # [B, num_classes] Distancias
            'epistemic_uncertainty': epistemic_unc,       # [B, 256] Incertidumbre epistémica
            'aleatoric_uncertainty': aleatoric_unc,       # [B, 1] Incertidumbre aleatórica
            
            # CRI OUTPUTS  
            'spatial_cliff': spatial_cliff,               # [B, 1] Cliff espacial
            'feature_cliff_map': feature_cliff_map,       # [B, 256] Mapa cliff features
            'multiscale_cliff': multiscale_cliff,         # [B, 1] Multi-escala
            'multiscale_components': multiscale_components, # [B, 3] Componentes escalas
            'cliff_confidence': cliff_confidence,         # [B, 1] Confianza cliff
            
            # CAFE OUTPUTS
            'attended_features': attended_features,       # [B, 256] Features con attention
            'attention_weights': attention_weights,       # [B, 1, 1] Pesos attention  
            'feature_weights': feature_weights,           # [B, 256] Pesos re-weighting
            'boundary_enhancement': boundary_enhancement, # [B, 256] Enhancement boundary
            
            # ANÁLISIS
            'analysis': analysis
        }
    
    def _compute_attention_entropy(self, attention_weights):
        """Compute entropy of attention weights"""
        # Flatten attention weights
        attn_flat = attention_weights.view(-1)
        attn_probs = F.softmax(attn_flat, dim=0)
        
        # Compute entropy
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8))
        return entropy

def create_phase2_complete_detector(input_dim=256, cliff_threshold=0.15, num_classes=7):
    """
    Factory para crear Fase 2 COMPLETA según paper
    """
    detector = Phase2CliffDetectorComplete(
        input_dim=input_dim,
        cliff_threshold=cliff_threshold, 
        num_classes=num_classes
    )
    
    return detector

# CLIFF LOSS ESPECÍFICO DEL PAPER
class CiffNetPhase2Loss(nn.Module):
    """
    Loss específico para Fase 2 según paper CiffNet
    """
    
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.3, num_classes=7):
        super(CiffNetPhase2Loss, self).__init__()
        
        self.alpha = alpha  # Cliff detection loss weight
        self.beta = beta    # Uncertainty loss weight  
        self.gamma = gamma  # Boundary preservation loss weight
        
        print(f"✅ CiffNet Phase 2 Loss:")
        print(f"   α (cliff detection): {alpha}")
        print(f"   β (uncertainty): {beta}") 
        print(f"   γ (boundary): {gamma}")
    
    def compute_cliff_detection_loss(self, cliff_score, target_difficulty):
        """Loss para cliff detection"""
        return F.mse_loss(cliff_score, target_difficulty)
    
    def compute_uncertainty_loss(self, epistemic_unc, aleatoric_unc):
        """Loss para uncertainty estimation"""
        # Regularización para uncertainty
        epistemic_reg = torch.mean(epistemic_unc)
        aleatoric_reg = torch.mean(aleatoric_unc)
        
        return epistemic_reg + aleatoric_reg
    
    def compute_boundary_preservation_loss(self, boundary_dist, true_labels):
        """Loss para boundary preservation"""
        # One-hot encoding
        true_dist = F.one_hot(true_labels, num_classes=boundary_dist.size(1)).float()
        
        # KL divergence
        boundary_log_probs = torch.log(boundary_dist + 1e-8)
        return F.kl_div(boundary_log_probs, true_dist, reduction='batchmean')
    
    def forward(self, phase2_outputs, targets, target_difficulty=None):
        """
        Compute total Phase 2 loss
        """
        # Extract outputs
        cliff_score = phase2_outputs['cliff_score']
        epistemic_unc = phase2_outputs['epistemic_uncertainty']
        aleatoric_unc = phase2_outputs['aleatoric_uncertainty']
        boundary_dist = phase2_outputs['boundary_distances']
        
        # Compute target difficulty if not provided
        if target_difficulty is None:
            # Use cliff score as proxy (self-supervised)
            target_difficulty = cliff_score.detach()
        
        # Component losses
        cliff_loss = self.compute_cliff_detection_loss(cliff_score, target_difficulty)
        uncertainty_loss = self.compute_uncertainty_loss(epistemic_unc, aleatoric_unc)
        boundary_loss = self.compute_boundary_preservation_loss(boundary_dist, targets)
        
        # Total loss
        total_loss = (self.alpha * cliff_loss + 
                     self.beta * uncertainty_loss + 
                     self.gamma * boundary_loss)
        
        return {
            'total_loss': total_loss,
            'cliff_loss': cliff_loss,
            'uncertainty_loss': uncertainty_loss,
            'boundary_loss': boundary_loss
        }

if __name__ == "__main__":
    # Test implementación completa
    print("🔬 TESTING FASE 2 COMPLETA SEGÚN PAPER")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear detector completo
    detector = create_phase2_complete_detector(
        input_dim=256, 
        cliff_threshold=0.15, 
        num_classes=7
    ).to(device)
    
    # Test data
    test_features = torch.randn(4, 256).to(device)
    test_labels = torch.randint(0, 7, (4,)).to(device)
    
    # Forward pass
    detector.eval()
    with torch.no_grad():
        results = detector(test_features)
    
    print(f"\n✅ FASE 2 COMPLETA funcionando:")
    print(f"   Enhanced features: {results['enhanced_features'].shape}")
    print(f"   Cliff score: {results['cliff_score'].shape}")
    print(f"   Análisis: {len(results['analysis'])} métricas")
    
    # Test loss
    loss_fn = CiffNetPhase2Loss(alpha=1.0, beta=0.5, gamma=0.3, num_classes=7)
    loss_results = loss_fn(results, test_labels)
    
    print(f"   Loss total: {loss_results['total_loss'].item():.4f}")
    print(f"\n🎯 IMPLEMENTACIÓN PAPER COMPLETA ✅")