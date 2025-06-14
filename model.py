import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

class MultiKernelSelfAttention(nn.Module):
    """Multi-Kernel Self-Attention (MKSA) con dimensiones dinámicas"""
    def __init__(self, in_channels, reduction=8, kernel_sizes=[1, 3, 5, 7]):
        super(MultiKernelSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction, 8)  # Mínimo 8 canales
        self.kernel_sizes = kernel_sizes
        
        # Multi-queries con diferentes kernel sizes
        self.multi_queries = nn.ModuleList()
        for k_size in kernel_sizes:
            padding = k_size // 2
            self.multi_queries.append(
                nn.Conv2d(in_channels, self.reduced_channels, 
                         kernel_size=k_size, padding=padding, bias=False)
            )
        
        # Key y Value compartidos
        self.key = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Proyección final
        self.projection = nn.Conv2d(in_channels * len(kernel_sizes), in_channels, 1, bias=False)
        
        # Normalización
        self.norm = nn.BatchNorm2d(in_channels)
        
        # Activación suave
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generar múltiples queries
        queries = []
        for i, query_conv in enumerate(self.multi_queries):
            q = query_conv(x)  # [B, reduced_channels, H, W]
            q = q.view(B, self.reduced_channels, -1)  # [B, reduced_channels, H*W]
            queries.append(q)
        
        # Key y Value
        k = self.key(x).view(B, self.reduced_channels, -1)  # [B, reduced_channels, H*W]
        v = self.value(x).view(B, C, -1)  # [B, C, H*W]
        
        # Compute attention para cada query
        attended_features = []
        for q in queries:
            # Attention scores
            attention_scores = torch.bmm(q.transpose(1, 2), k)  # [B, H*W, H*W]
            attention_weights = self.softmax(attention_scores / math.sqrt(self.reduced_channels))
            
            # Apply attention
            attended = torch.bmm(v, attention_weights.transpose(1, 2))  # [B, C, H*W]
            attended = attended.view(B, C, H, W)  # [B, C, H, W]
            attended_features.append(attended)
        
        # Concatenar y proyectar
        concatenated = torch.cat(attended_features, dim=1)  # [B, C*len(kernels), H, W]
        output = self.projection(concatenated)  # [B, C, H, W]
        
        # Residual connection + normalización
        output = self.norm(output + x)
        
        return output

class CrossStageAttention(nn.Module):
    """Cross-Stage Attention mejorado"""
    def __init__(self, feature_dims, final_dim=1280):
        super(CrossStageAttention, self).__init__()
        self.feature_dims = feature_dims
        self.final_dim = final_dim
        
        # Proyecciones para igualar dimensiones
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            self.projections.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dim, final_dim // len(feature_dims), 1),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(final_dim, len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
    def forward(self, feature_maps):
        # Proyectar cada feature map
        projected = []
        for i, feature_map in enumerate(feature_maps):
            proj = self.projections[i](feature_map)  # [B, final_dim//len, 1, 1]
            projected.append(proj.squeeze(-1).squeeze(-1))  # [B, final_dim//len]
        
        # Concatenar características
        concatenated = torch.cat(projected, dim=1)  # [B, final_dim]
        
        # Calcular attention weights
        weights = self.attention_weights(concatenated)  # [B, len(feature_dims)]
        
        # Aplicar attention
        weighted_features = []
        for i, proj_feat in enumerate(projected):
            weighted = proj_feat * weights[:, i:i+1]
            weighted_features.append(weighted)
        
        return torch.cat(weighted_features, dim=1)  # [B, final_dim]

class CIFFNetPhase1(nn.Module):
    """CIFF-Net Fase 1 con dimensiones dinámicas"""
    def __init__(self, num_classes=7, backbone='efficientnet_b0', pretrained=True):
        super(CIFFNetPhase1, self).__init__()
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        
        # Obtener información del backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_info = [(f.shape[1], f.shape[2:]) for f in features]
        
        print(f"📏 Feature dimensions: {self.feature_info}")
        
        # MKSA modules para cada nivel de características
        self.mksa_modules = nn.ModuleList()
        for channels, spatial_size in self.feature_info:
            self.mksa_modules.append(
                MultiKernelSelfAttention(channels, reduction=8)
            )
        
        # Cross-Stage Attention
        feature_dims = [info[0] for info in self.feature_info]
        final_feature_dim = sum(channels // len(feature_dims) for channels in feature_dims)
        
        self.cs_attention = CrossStageAttention(feature_dims, final_feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Clasificador
        classifier_input_dim = final_feature_dim * 2  # avg + max pooling
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(classifier_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Para almacenar feature maps (útil para visualización)
        self.feature_maps = []
        
    def forward(self, x):
        # Limpiar feature maps anteriores
        self.feature_maps = []
        
        # Extraer características del backbone
        features = self.backbone(x)
        
        # Aplicar MKSA a cada nivel
        attended_features = []
        for i, feature_map in enumerate(features):
            self.feature_maps.append(feature_map)
            attended = self.mksa_modules[i](feature_map)
            attended_features.append(attended)
        
        # Cross-stage attention
        fused_features = self.cs_attention(attended_features)
        
        # Clasificación
        output = self.classifier(fused_features)
        
        return output
    
    def get_attention_maps(self, x):
        """Obtener mapas de atención para visualización"""
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)
            return self.feature_maps

def create_ciff_net_phase1(num_classes=7, backbone='efficientnet_b0', pretrained=True):
    """Factory function para crear CIFF-Net Fase 1"""
    return CIFFNetPhase1(num_classes, backbone, pretrained)

def model_summary(model, input_size=(1, 3, 224, 224)):
    """Resumen mejorado del modelo"""
    print("=" * 60)
    print("CIFF-NET FASE 1 - MODEL SUMMARY")
    print("=" * 60)
    
    # Información básica
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    
    # Test forward pass
    model.eval()
    try:
        with torch.no_grad():
            x = torch.randn(*input_size)
            output = model(x)
            print(f"Input shape: {tuple(x.shape)}")
            print(f"Output shape: {tuple(output.shape)}")
            
            # Información de características
            if hasattr(model, 'feature_info'):
                print(f"\nFeature levels: {len(model.feature_info)}")
                for i, (channels, spatial) in enumerate(model.feature_info):
                    print(f"  Level {i+1}: {channels} channels, spatial {spatial}")
        
        print("✅ Forward pass exitoso")
        
    except Exception as e:
        print(f"❌ Error en forward pass: {e}")
        # Información de debug
        if hasattr(model, 'backbone'):
            print(f"Backbone: {type(model.backbone)}")
        
    print("=" * 60)

def count_parameters(model):
    """Contar parámetros del modelo"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Desglose por módulos
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    mksa_params = sum(p.numel() for p in model.mksa_modules.parameters())
    cs_attention_params = sum(p.numel() for p in model.cs_attention.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'backbone': backbone_params,
        'mksa': mksa_params,
        'cs_attention': cs_attention_params,
        'classifier': classifier_params
    }

if __name__ == "__main__":
    print("🧠 Probando CIFF-Net Fase 1...")
    
    # Crear modelo
    model = create_ciff_net_phase1(num_classes=7)
    
    # Resumen
    model_summary(model)
    
    # Contar parámetros
    params = count_parameters(model)
    print("\n📊 Desglose de parámetros:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test con batch
    print("\n🔧 Probando con batch...")
    model.eval()
    with torch.no_grad():
        batch = torch.randn(4, 3, 224, 224)
        output = model(batch)
        print(f"✅ Batch output: {output.shape}")
        
        # Obtener attention maps
        attention_maps = model.get_attention_maps(batch[:1])
        print(f"📍 Attention maps: {len(attention_maps)} niveles")
        for i, am in enumerate(attention_maps):
            print(f"  Nivel {i+1}: {am.shape}")