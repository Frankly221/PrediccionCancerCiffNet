import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from torch.utils.checkpoint import checkpoint
import numpy as np

class FocalLoss(nn.Module):
    """Focal Loss para clases desbalanceadas"""
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedMKSA(nn.Module):
    """MKSA mejorado con múltiples escalas"""
    def __init__(self, in_channels, reduction=8, num_heads=4):
        super(ImprovedMKSA, self).__init__()
        self.in_channels = in_channels
        self.reduced_channels = max(in_channels // reduction, 16)
        self.num_heads = num_heads
        self.head_dim = self.reduced_channels // num_heads
        
        # Multi-head attention
        self.query = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False)
        self.value = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        
        # Multi-scale convolutions
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.Conv2d(in_channels, in_channels // 4, 5, padding=2),
            nn.Conv2d(in_channels, in_channels // 4, 7, padding=3)
        ])
        
        # Channel attention mejorado
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention mejorado
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 16, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Multi-scale features
        multi_scale_features = []
        for conv in self.multi_scale:
            multi_scale_features.append(conv(x))
        
        multi_scale_out = torch.cat(multi_scale_features, dim=1)
        
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_concat = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_concat)
        x_spatial = x_channel * spatial_att
        
        # Combine all features
        output = multi_scale_out + x_spatial
        output = self.norm1(output + identity)
        
        return self.dropout(output)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ImprovedCrossStageAttention(nn.Module):
    """Cross-stage attention mejorado"""
    def __init__(self, feature_dims, output_dim=1024):
        super(ImprovedCrossStageAttention, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Proyecciones adaptativas
        self.projections = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        
        proj_dim_per_level = output_dim // len(feature_dims)
        
        for i, dim in enumerate(feature_dims):
            # Proyección con más capacidad
            self.projections.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(dim, proj_dim_per_level * 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(proj_dim_per_level * 2, proj_dim_per_level),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2)
                )
            )
            
            # SE blocks para features espaciales
            self.se_blocks.append(SEBlock(dim))
        
        # Attention entre niveles
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=proj_dim_per_level,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusión final
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, feature_maps):
        # Aplicar SE blocks y proyecciones
        features = []
        for i, feature_map in enumerate(feature_maps):
            # SE enhancement
            enhanced = self.se_blocks[i](feature_map)
            # Projection
            feat = self.projections[i](enhanced)
            features.append(feat)
        
        # Stack para attention
        stacked_features = torch.stack(features, dim=1)  # [B, num_levels, proj_dim]
        
        # Cross attention entre niveles
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Concatenar y fusionar
        concatenated = attended_features.flatten(1)  # [B, output_dim]
        return self.fusion(concatenated)

class ImprovedCIFFNet(nn.Module):
    """CIFF-Net mejorado para mejor rendimiento"""
    def __init__(self, num_classes=7, backbone='efficientnet_b1', pretrained=True):
        super(ImprovedCIFFNet, self).__init__()
        
        print(f"🚀 Creando CIFF-Net MEJORADO...")
        print(f"   Backbone: {backbone}")
        print(f"   Clases: {num_classes}")
        
        # Backbone más potente
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True)
        
        # Feature info
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.backbone = self.backbone.cuda()
            
            features = self.backbone(dummy_input)
            self.feature_info = [(f.shape[1], f.shape[2:]) for f in features]
        
        print(f"🔥 Feature dimensions: {self.feature_info}")
        
        # MKSA mejorado en múltiples niveles
        self.mksa_modules = nn.ModuleList()
        for i, (channels, spatial_size) in enumerate(self.feature_info):
            if i >= len(self.feature_info) - 2:  # Últimas 2 capas
                self.mksa_modules.append(
                    ImprovedMKSA(channels, reduction=8, num_heads=4)
                )
                print(f"   Level {i}: ImprovedMKSA aplicado")
            else:
                # SE blocks para otras capas
                self.mksa_modules.append(SEBlock(channels))
                print(f"   Level {i}: SE Block aplicado")
        
        # Cross-stage attention mejorado
        feature_dims = [info[0] for info in self.feature_info]
        self.cs_attention = ImprovedCrossStageAttention(feature_dims, output_dim=1024)
        
        # Clasificador más sofisticado
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
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
        
        # Gradient checkpointing
        self.use_checkpoint = True
        
        print(f"✅ CIFF-Net MEJORADO creado exitosamente!")
        
    def forward(self, x):
        # Extraer características
        if self.training and self.use_checkpoint:
            features = checkpoint(self.backbone, x)
        else:
            features = self.backbone(x)
        
        # Aplicar attention mejorado
        attended_features = []
        for i, feature_map in enumerate(features):
            if isinstance(self.mksa_modules[i], ImprovedMKSA):
                if self.training and self.use_checkpoint:
                    attended = checkpoint(self.mksa_modules[i], feature_map)
                else:
                    attended = self.mksa_modules[i](feature_map)
            else:
                attended = self.mksa_modules[i](feature_map)
            
            attended_features.append(attended)
        
        # Cross-stage attention
        fused_features = self.cs_attention(attended_features)
        
        return self.classifier(fused_features)

def create_improved_ciff_net(num_classes=7, backbone='efficientnet_b1', pretrained=True):
    """Factory para modelo mejorado"""
    return ImprovedCIFFNet(num_classes, backbone, pretrained)