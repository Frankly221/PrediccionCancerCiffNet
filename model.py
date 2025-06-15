import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch.utils.checkpoint import checkpoint

class MemoryEfficientMKSA(nn.Module):
    """MKSA ultra-eficiente para RTX 3070 Ti (8GB)"""
    def __init__(self, in_channels, reduction=16, max_spatial_size=16):
        super(MemoryEfficientMKSA, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction, 8)
        self.max_spatial_size = max_spatial_size
        
        # Solo 2 kernels para ahorrar memoria
        self.query = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Channel attention como alternativa eficiente
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention simple
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Si la resolución es muy alta, usar solo channel+spatial attention
        if H * W > self.max_spatial_size * self.max_spatial_size:
            return self._lightweight_attention(x)
        
        # Self-attention solo para resoluciones pequeñas
        return self._self_attention(x)
    
    def _lightweight_attention(self, x):
        """Attention ligero para resoluciones altas"""
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = torch.sigmoid(self.spatial_conv(spatial_concat))
        
        output = x_channel * spatial_weights
        return self.norm(output + x)
    
    def _self_attention(self, x):
        """Self-attention para resoluciones pequeñas"""
        B, C, H, W = x.shape
        
        # Downsample agresivamente si es necesario
        if H > self.max_spatial_size or W > self.max_spatial_size:
            scale = min(self.max_spatial_size / H, self.max_spatial_size / W)
            x_small = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            x_small = x
        
        B, C, H_s, W_s = x_small.shape
        
        # QKV
        q = self.query(x_small).view(B, self.reduced_channels, -1)  # [B, C_r, H_s*W_s]
        k = self.key(x_small).view(B, self.reduced_channels, -1)    # [B, C_r, H_s*W_s]
        v = self.value(x_small).view(B, C, -1)                      # [B, C, H_s*W_s]
        
        # Attention
        attention_scores = torch.bmm(q.transpose(1, 2), k) / math.sqrt(self.reduced_channels)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(v, attention_weights.transpose(1, 2))
        attended = attended.view(B, C, H_s, W_s)
        
        # Upsample back
        if (H_s, W_s) != (H, W):
            attended = F.interpolate(attended, size=(H, W), mode='bilinear', align_corners=False)
        
        return self.norm(attended + x)

class CompactCrossStageAttention(nn.Module):
    """Cross-stage attention compacto"""
    def __init__(self, feature_dims, output_dim=512):
        super(CompactCrossStageAttention, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Proyecciones simples
        self.projections = nn.ModuleList()
        for dim in feature_dims:
            self.projections.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(dim, output_dim // len(feature_dims)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
            )
        
        # Fusión simple
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, feature_maps):
        features = []
        for i, feature_map in enumerate(feature_maps):
            feat = self.projections[i](feature_map)
            features.append(feat)
        
        concatenated = torch.cat(features, dim=1)
        return self.fusion(concatenated)

class CIFFNetRTX8GB(nn.Module):
    """CIFF-Net optimizado para 8GB VRAM"""
    def __init__(self, num_classes=7, backbone='efficientnet_b0', pretrained=True):
        super(CIFFNetRTX8GB, self).__init__()
        
        # Backbone más conservador
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        
        # Feature info
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.backbone = self.backbone.cuda()
            
            features = self.backbone(dummy_input)
            self.feature_info = [(f.shape[1], f.shape[2:]) for f in features]
        
        print(f"🔥 RTX 8GB Feature dimensions: {self.feature_info}")
        
        # MKSA solo en la última capa
        self.mksa_modules = nn.ModuleList()
        for i, (channels, spatial_size) in enumerate(self.feature_info):
            if i == len(self.feature_info) - 1:  # Solo última capa
                max_spatial = min(max(spatial_size), 16)  # Limitar resolución
                self.mksa_modules.append(
                    MemoryEfficientMKSA(channels, reduction=16, max_spatial_size=max_spatial)
                )
            else:
                # Channel attention simple para otras capas
                self.mksa_modules.append(
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(channels, channels // 16, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(channels // 16, channels, 1),
                        nn.Sigmoid()
                    )
                )
        
        # Cross-stage attention compacto
        feature_dims = [info[0] for info in self.feature_info]
        self.cs_attention = CompactCrossStageAttention(feature_dims, output_dim=512)
        
        # Clasificador simple
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Gradient checkpointing para ahorrar memoria
        self.use_checkpoint = True
        
    def forward(self, x):
        # Extraer características con checkpointing
        if self.training and self.use_checkpoint:
            features = checkpoint(self.backbone, x)
        else:
            features = self.backbone(x)
        
        # Aplicar attention
        attended_features = []
        for i, feature_map in enumerate(features):
            if isinstance(self.mksa_modules[i], MemoryEfficientMKSA):
                # MKSA con checkpointing
                if self.training and self.use_checkpoint:
                    attended = checkpoint(self.mksa_modules[i], feature_map)
                else:
                    attended = self.mksa_modules[i](feature_map)
            else:
                # Channel attention simple
                channel_attention = self.mksa_modules[i](feature_map)
                attended = feature_map * channel_attention
            
            attended_features.append(attended)
        
        # Cross-stage attention
        fused_features = self.cs_attention(attended_features)
        
        return self.classifier(fused_features)

def create_ciff_net_rtx8gb(num_classes=7, backbone='efficientnet_b0', pretrained=True):
    """Factory para RTX 8GB"""
    return CIFFNetRTX8GB(num_classes, backbone, pretrained)

def rtx8gb_model_summary(model, input_size=(1, 3, 224, 224)):
    """Resumen RTX 8GB"""
    print("=" * 60)
    print("🔥 CIFF-NET RTX 8GB MEMORY-OPTIMIZED")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {total_params:,}")
    print(f"Memoria estimada: {total_params * 4 / 1e9:.2f} GB")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            # Limpiar cache
            torch.cuda.empty_cache()
            
            x = torch.randn(*input_size).cuda()
            memory_before = torch.cuda.memory_allocated() / 1e9
            
            output = model(x)
            
            memory_after = torch.cuda.memory_allocated() / 1e9
            memory_used = memory_after - memory_before
            
            print(f"Input: {tuple(x.shape)} -> Output: {tuple(output.shape)}")
            print(f"VRAM utilizada: {memory_used:.2f} GB")
            print(f"VRAM libre: {8.0 - memory_after:.2f} GB")
    
    print("=" * 60)

if __name__ == "__main__":
    print("🔥 Probando CIFF-Net RTX 8GB...")
    
    if torch.cuda.is_available():
        # Limpiar cache
        torch.cuda.empty_cache()
        
        model = create_ciff_net_rtx8gb(num_classes=7)
        rtx8gb_model_summary(model)
    else:
        print("❌ CUDA no disponible")