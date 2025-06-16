import torch
import torch.nn as nn
import timm
import numpy as np
from torch.cuda.amp import autocast

class Phase1FeatureExtractor(nn.Module):
    """
    FASE 1 del paper CiffNet: Feature Extraction
    Optimizada para RTX 3070 con todas las mejoras de velocidad
    """
    
    def __init__(self, backbone='efficientnet_b1', pretrained=True):
        super(Phase1FeatureExtractor, self).__init__()
        
        print(f"🔧 FASE 1 - Feature Extraction inicializando...")
        print(f"   Backbone: {backbone}")
        print(f"   Pretrained: {pretrained}")
        
        # BACKBONE OPTIMIZADO (como en paper)
        self.backbone = timm.create_model(
            backbone, 
            pretrained=pretrained,
            features_only=True,           # Solo features, sin clasificador
            out_indices=[1, 2, 3, 4],     # Multi-scale según paper
            drop_rate=0.2,                # Dropout interno
            drop_path_rate=0.1            # Stochastic depth
        )
        
        # DETECTAR DIMENSIONES AUTOMÁTICAMENTE
        self._detect_feature_dimensions()
        
        # FEATURE PROCESSORS (Según paper)
        self.feature_processors = nn.ModuleList()
        
        for i, dim in enumerate(self.feature_dims):
            processor = nn.Sequential(
                # Convolución 1x1 para reducir dimensiones
                nn.Conv2d(dim, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Global Average Pooling (paper específico)
                nn.AdaptiveAvgPool2d(1),
                
                # Flatten para concatenar
                nn.Flatten()
            )
            self.feature_processors.append(processor)
        
        # FEATURE FUSION LAYER (del paper)
        total_features = 256 * len(self.feature_dims)  # 256 * 4 = 1024
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        # OPTIMIZACIONES RTX 3070
        self._apply_optimizations()
        
        print(f"✅ FASE 1 creada:")
        print(f"   Feature dims: {self.feature_dims}")
        print(f"   Total params: {sum(p.numel() for p in self.parameters()):,}")
        print(f"   Output dim: 256 (listo para Fase 2)")
        
    def _detect_feature_dimensions(self):
        """Detectar dimensiones del backbone automáticamente"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dims = [f.shape[1] for f in features]
            
        print(f"   Dimensiones detectadas: {self.feature_dims}")
    
    def _apply_optimizations(self):
        """Aplicar optimizaciones RTX 3070"""
        # Channels last para Ampere
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module = module.to(memory_format=torch.channels_last)
        
        print(f"   ⚡ Optimizaciones RTX 3070 aplicadas")
    
    def forward(self, x):
        """
        Forward pass FASE 1 - Feature Extraction
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            dict: {
                'multi_scale_features': [B, 1024] - Features concatenadas
                'fused_features': [B, 256] - Features fusionadas para Fase 2
                'individual_features': [feat1, feat2, feat3, feat4] - Para análisis
            }
        """
        batch_size = x.size(0)
        
        # Asegurar channels last
        if x.stride()[1] == 1:  # Ya en channels last
            backbone_input = x
        else:
            backbone_input = x.to(memory_format=torch.channels_last)
        
        # EXTRACT MULTI-SCALE FEATURES (paper step 1)
        with autocast():
            backbone_features = self.backbone(backbone_input)
        
        # PROCESS EACH SCALE (paper step 2)
        processed_features = []
        
        for i, (feat, processor) in enumerate(zip(backbone_features, self.feature_processors)):
            # Aplicar processor específico por escala
            processed = processor(feat)  # [B, 256]
            processed_features.append(processed)
        
        # CONCATENATE ALL SCALES (paper step 3)
        multi_scale_features = torch.cat(processed_features, dim=1)  # [B, 1024]
        
        # FEATURE FUSION (paper step 4)
        fused_features = self.feature_fusion(multi_scale_features)  # [B, 256]
        
        return {
            'multi_scale_features': multi_scale_features,  # Para Fase 2
            'fused_features': fused_features,              # Para Fase 2  
            'individual_features': backbone_features,      # Para análisis
            'processed_features': processed_features       # Para debug
        }
    
    def get_feature_info(self):
        """Información sobre las features extraídas"""
        return {
            'backbone': str(self.backbone),
            'feature_dims': self.feature_dims,
            'output_dim': 256,
            'multi_scale_dim': sum([256] * len(self.feature_dims)),
            'scales': len(self.feature_dims)
        }

def create_phase1_extractor(backbone='efficientnet_b1', pretrained=True):
    """
    Factory function para crear Phase 1 optimizada
    """
    extractor = Phase1FeatureExtractor(
        backbone=backbone,
        pretrained=pretrained
    )
    
    return extractor

# FUNCIONES DE TESTING Y ANÁLISIS
def test_phase1_extraction(extractor, sample_input):
    """Test de la Fase 1 con análisis detallado"""
    print(f"\n🧪 TESTING FASE 1:")
    print(f"   Input shape: {sample_input.shape}")
    
    extractor.eval()
    with torch.no_grad():
        results = extractor(sample_input)
    
    print(f"✅ RESULTADOS FASE 1:")
    print(f"   Multi-scale features: {results['multi_scale_features'].shape}")
    print(f"   Fused features: {results['fused_features'].shape}")
    print(f"   Individual features: {[f.shape for f in results['individual_features']]}")
    
    return results

def analyze_phase1_features(results):
    """Análisis de las features de Fase 1"""
    multi_scale = results['multi_scale_features']
    fused = results['fused_features']
    
    analysis = {
        'multi_scale_stats': {
            'mean': float(multi_scale.mean()),
            'std': float(multi_scale.std()),
            'min': float(multi_scale.min()),
            'max': float(multi_scale.max())
        },
        'fused_stats': {
            'mean': float(fused.mean()),
            'std': float(fused.std()),
            'min': float(fused.min()),
            'max': float(fused.max())
        },
        'feature_diversity': float(torch.std(fused, dim=1).mean()),
        'activation_sparsity': float((fused == 0).float().mean())
    }
    
    print(f"\n📊 ANÁLISIS FEATURES FASE 1:")
    print(f"   Multi-scale: μ={analysis['multi_scale_stats']['mean']:.3f}, σ={analysis['multi_scale_stats']['std']:.3f}")
    print(f"   Fused: μ={analysis['fused_stats']['mean']:.3f}, σ={analysis['fused_stats']['std']:.3f}")
    print(f"   Diversidad: {analysis['feature_diversity']:.3f}")
    print(f"   Sparsity: {analysis['activation_sparsity']*100:.1f}%")
    
    return analysis

if __name__ == "__main__":
    # Test básico de Fase 1
    print("🔬 TESTING FASE 1 - FEATURE EXTRACTION")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear extractor
    extractor = create_phase1_extractor('efficientnet_b1', True)
    extractor = extractor.to(device)
    
    # Test input
    test_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test extraction
    results = test_phase1_extraction(extractor, test_input)
    
    # Análisis
    analysis = analyze_phase1_features(results)
    
    print(f"\n✅ FASE 1 funcionando correctamente")
    print(f"📋 Listo para implementar Fase 2 (Cliff Detection)")