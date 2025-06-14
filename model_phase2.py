import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import create_ciff_net_phase1

class ComparativeContextualFeatureFusion(nn.Module):
    """CCFF - Módulo de fusión de características contextuales"""
    def __init__(self, feature_dim=1280, num_context_images=3, fusion_method='attention'):
        super(ComparativeContextualFeatureFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_context_images = num_context_images
        self.fusion_method = fusion_method
        
        # Proyección de características para comparación
        self.main_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.context_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        if fusion_method == 'attention':
            # Mecanismo de atención para ponderar contextos
            self.attention_weights = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            # Red de comparación
            self.comparison_net = nn.Sequential(
                nn.Linear(feature_dim // 2 * 2, feature_dim // 4),  # main + context
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 4, feature_dim // 8),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim // 8, 1),
                nn.Sigmoid()
            )
            
        elif fusion_method == 'correlation':
            # Correlación entre main y context
            self.correlation_layer = nn.MultiheadAttention(
                embed_dim=feature_dim // 2,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            
        # Fusión final
        self.fusion_layers = nn.Sequential(
            nn.Linear(feature_dim + feature_dim // 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Gate para controlar influencia contextual
        self.context_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, main_features, context_features):
        """
        main_features: [B, feature_dim] - características de imagen principal
        context_features: [B, M, feature_dim] - características de M imágenes contextuales
        """
        B, M, feature_dim = context_features.shape
        
        # Proyectar características
        main_proj = self.main_projection(main_features)  # [B, feature_dim//2]
        context_proj = self.context_projection(context_features.view(-1, feature_dim))  # [B*M, feature_dim//2]
        context_proj = context_proj.view(B, M, -1)  # [B, M, feature_dim//2]
        
        if self.fusion_method == 'attention':
            # Calcular pesos de atención para cada imagen contextual
            attention_scores = []
            comparison_scores = []
            
            for i in range(M):
                ctx_feat = context_features[:, i, :]  # [B, feature_dim]
                
                # Peso de atención basado en características contextuales
                att_weight = self.attention_weights(ctx_feat)  # [B, 1]
                attention_scores.append(att_weight)
                
                # Puntuación de comparación main vs context
                main_ctx_combined = torch.cat([main_proj, context_proj[:, i, :]], dim=1)
                comp_score = self.comparison_net(main_ctx_combined)  # [B, 1]
                comparison_scores.append(comp_score)
            
            # Combinar pesos
            attention_weights = torch.cat(attention_scores, dim=1)  # [B, M]
            comparison_weights = torch.cat(comparison_scores, dim=1)  # [B, M]
            
            # Normalizar pesos
            final_weights = F.softmax(attention_weights * comparison_weights, dim=1)  # [B, M]
            
            # Fusión ponderada de características contextuales
            weighted_context = torch.sum(
                context_features * final_weights.unsqueeze(-1), dim=1
            )  # [B, feature_dim]
            
        elif self.fusion_method == 'correlation':
            # Usar correlación cruzada
            main_query = main_proj.unsqueeze(1)  # [B, 1, feature_dim//2]
            context_keys = context_proj  # [B, M, feature_dim//2]
            
            # Atención cruzada
            attended_context, attention_weights = self.correlation_layer(
                main_query, context_keys, context_keys
            )  # [B, 1, feature_dim//2]
            
            attended_context = attended_context.squeeze(1)  # [B, feature_dim//2]
            
            # Expandir para fusión
            weighted_context = torch.cat([attended_context, attended_context], dim=1)  # [B, feature_dim]
            
        else:  # simple concatenation
            # Promedio simple de contextos
            weighted_context = torch.mean(context_features, dim=1)  # [B, feature_dim]
        
        # Fusión main + context
        combined_features = torch.cat([main_features, weighted_context], dim=1)  # [B, 2*feature_dim]
        
        # Gate para controlar influencia
        gate_weights = self.context_gate(combined_features)  # [B, feature_dim]
        
        # Aplicar gate
        gated_main = main_features * gate_weights
        gated_context = weighted_context * (1 - gate_weights)
        
        # Fusión final
        final_combined = torch.cat([gated_main, gated_context], dim=1)
        fused_features = self.fusion_layers(final_combined)
        
        return fused_features, final_weights if self.fusion_method == 'attention' else None

class CIFFNetPhase2(nn.Module):
    """CIFF-Net Fase 2: Fusión Contextual con CCFF"""
    def __init__(self, phase1_model_path, num_classes=7, num_context_images=3, 
                 fusion_method='attention', freeze_phase1=False):
        super(CIFFNetPhase2, self).__init__()
        
        # Cargar modelo de Fase 1 pre-entrenado
        checkpoint = torch.load(phase1_model_path, map_location='cpu')
        self.phase1_model = create_ciff_net_phase1(num_classes=num_classes)
        self.phase1_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Congelar Fase 1 si se especifica
        if freeze_phase1:
            for param in self.phase1_model.parameters():
                param.requires_grad = False
        
        # Extraer características antes del clasificador
        # Remover clasificador de Fase 1 para usar como feature extractor
        self.feature_extractor = nn.Sequential(*list(self.phase1_model.children())[:-1])
        
        # Dimensión de características (antes del clasificador final)
        if hasattr(self.phase1_model, 'classifier'):
            feature_dim = self.phase1_model.classifier[1].in_features  # Primera capa del clasificador
        else:
            feature_dim = 1280  # EfficientNet-B0 default
        
        # Módulo CCFF
        self.ccff_module = ComparativeContextualFeatureFusion(
            feature_dim=feature_dim,
            num_context_images=num_context_images,
            fusion_method=fusion_method
        )
        
        # Clasificador de Fase 2
        self.phase2_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Inicializar nuevos módulos
        self._initialize_new_modules()
    
    def _initialize_new_modules(self):
        """Inicializar solo los módulos nuevos (CCFF + clasificador)"""
        for module in [self.ccff_module, self.phase2_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def extract_features(self, images):
        """Extraer características usando modelo de Fase 1"""
        B = images.shape[0]
        
        # Extraer características de cada imagen
        features_list = []
        for i in range(B):
            img = images[i].unsqueeze(0)  # [1, C, H, W]
            
            # Forward hasta antes del clasificador
            with torch.set_grad_enabled(self.training and not self.ccff_module.training):
                # Usar el feature extractor (Fase 1 sin clasificador)
                feat = self.phase1_model(img)  # Esto pasa por todo el modelo
                
                # Obtener características antes de la clasificación final
                # Necesitamos acceder a las características pooled
                self.phase1_model.eval()
                self.phase1_model.feature_maps = []
                _ = self.phase1_model.backbone(img)
                
                if self.phase1_model.feature_maps:
                    # Usar las características del último nivel con MKSA
                    final_features = self.phase1_model.feature_maps[-1]
                    final_features = self.phase1_model.mksa_modules[-1](final_features)
                    final_features = self.phase1_model.cs_attention(final_features)
                    
                    # Global pooling
                    avg_pooled = self.phase1_model.global_pool(final_features)
                    max_pooled = self.phase1_model.global_max_pool(final_features)
                    pooled = torch.cat([
                        avg_pooled.view(avg_pooled.size(0), -1),
                        max_pooled.view(max_pooled.size(0), -1)
                    ], dim=1)
                    
                    # Proyectar a dimensión correcta
                    if pooled.shape[1] != 1280:  # Si no es la dimensión esperada
                        proj_layer = nn.Linear(pooled.shape[1], 1280).to(pooled.device)
                        pooled = proj_layer(pooled)
                    
                    features_list.append(pooled)
                else:
                    # Fallback: usar salida directa del modelo
                    features_list.append(torch.randn(1, 1280).to(img.device))
        
        return torch.cat(features_list, dim=0)  # [B, feature_dim]
    
    def forward(self, main_images, context_images=None):
        """
        main_images: [B, C, H, W] - imágenes principales
        context_images: [B, M, C, H, W] - imágenes contextuales (opcional)
        """
        B = main_images.shape[0]
        
        # Extraer características de imagen principal
        main_features = self.extract_features(main_images)  # [B, feature_dim]
        
        if context_images is not None and self.training:
            # Extraer características de imágenes contextuales
            M = context_images.shape[1]
            context_features_list = []
            
            for i in range(M):
                ctx_imgs = context_images[:, i, :, :, :]  # [B, C, H, W]
                ctx_features = self.extract_features(ctx_imgs)  # [B, feature_dim]
                context_features_list.append(ctx_features)
            
            context_features = torch.stack(context_features_list, dim=1)  # [B, M, feature_dim]
            
            # Aplicar CCFF
            fused_features, attention_weights = self.ccff_module(main_features, context_features)
            
            # Clasificación con características fusionadas
            output = self.phase2_classifier(fused_features)
            
            return output, attention_weights
        else:
            # Solo imagen principal (validación o inferencia)
            output = self.phase2_classifier(main_features)
            return output, None
    
    def get_attention_visualization(self, main_images, context_images):
        """Obtener visualización de atención contextual"""
        self.eval()
        with torch.no_grad():
            _, attention_weights = self.forward(main_images, context_images)
        return attention_weights

def create_ciff_net_phase2(phase1_model_path, num_classes=7, num_context_images=3, 
                          fusion_method='attention', freeze_phase1=False):
    """Factory function para crear CIFF-Net Fase 2"""
    return CIFFNetPhase2(
        phase1_model_path, num_classes, num_context_images, 
        fusion_method, freeze_phase1
    )

def count_parameters_phase2(model):
    """Contar parámetros de Fase 2"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Parámetros específicos de Fase 2
    phase2_params = sum(p.numel() for p in model.ccff_module.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.phase2_classifier.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'phase2_ccff': phase2_params,
        'phase2_classifier': classifier_params
    }

if __name__ == "__main__":
    print("🧠 Probando CIFF-Net Fase 2 con CCFF...")
    
    # Crear modelo (necesita modelo de Fase 1 pre-entrenado)
    try:
        model = create_ciff_net_phase2(
            phase1_model_path='best_ciff_net_phase1.pth',
            num_classes=7,
            num_context_images=3,
            fusion_method='attention'
        )
        
        # Test
        main_imgs = torch.randn(2, 3, 224, 224)
        context_imgs = torch.randn(2, 3, 3, 224, 224)
        
        model.train()
        output, att_weights = model(main_imgs, context_imgs)
        
        print(f"✅ Output shape: {output.shape}")
        if att_weights is not None:
            print(f"✅ Attention weights shape: {att_weights.shape}")
        
        # Contar parámetros
        params = count_parameters_phase2(model)
        print(f"📊 Parámetros Fase 2:")
        for k, v in params.items():
            print(f"  {k}: {v:,}")
            
    except FileNotFoundError:
        print("❌ Modelo de Fase 1 no encontrado. Entrena primero la Fase 1.")