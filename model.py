import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B3_Weights, EfficientNet_B7_Weights

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=7, model_variant='b0', pretrained=True, dropout_rate=0.5):
        super(EfficientNetClassifier, self).__init__()
        
        self.model_variant = model_variant
        
        # Seleccionar variante de EfficientNet
        if model_variant == 'b0':
            if pretrained:
                self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b0(weights=None)
            feature_dim = 1280
            
        elif model_variant == 'b3':
            if pretrained:
                self.backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b3(weights=None)
            feature_dim = 1536
            
        elif model_variant == 'b7':
            if pretrained:
                self.backbone = models.efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.efficientnet_b7(weights=None)
            feature_dim = 2560
        
        # Reemplazar clasificador
        # EfficientNet tiene estructura: features + avgpool + classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        # Inicializar pesos de las nuevas capas
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializar pesos de las capas personalizadas"""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self, freeze=True):
        """Congelar/descongelar backbone para fine-tuning gradual"""
        for param in self.backbone.features.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_last_n_blocks(self, n_blocks=2):
        """Descongelar últimos n bloques para fine-tuning gradual"""
        total_blocks = len(list(self.backbone.features.children()))
        
        for i, child in enumerate(self.backbone.features.children()):
            if i >= total_blocks - n_blocks:
                for param in child.parameters():
                    param.requires_grad = True

class EfficientNetEnsemble(nn.Module):
    """Ensemble de múltiples EfficientNet como en el paper"""
    def __init__(self, num_classes=7, variants=['b0', 'b3'], pretrained=True):
        super(EfficientNetEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            EfficientNetClassifier(num_classes, variant, pretrained)
            for variant in variants
        ])
        
        # Capa de fusión opcional
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * len(variants), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.use_fusion = False  # Cambiar a True para usar fusión
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        if self.use_fusion:
            # Concatenar y pasar por capa de fusión
            combined = torch.cat(outputs, dim=1)
            return self.fusion(combined)
        else:
            # Promedio simple (como en muchos papers)
            return torch.stack(outputs).mean(dim=0)

def create_efficientnet_model(num_classes=7, variant='b0', pretrained=True):
    """Factory function para crear modelo EfficientNet individual"""
    return EfficientNetClassifier(num_classes, variant, pretrained)

def create_ensemble_model(num_classes=7, variants=['b0', 'b3'], pretrained=True):
    """Factory function para crear ensemble"""
    return EfficientNetEnsemble(num_classes, variants, pretrained)

def count_parameters(model):
    """Contar parámetros entrenables"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(1, 3, 224, 224)):
    """Resumen del modelo"""
    print(f"Modelo: {model.__class__.__name__}")
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(input_size)
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    # Prueba de modelos
    print("="*50)
    print("MODELOS EFFICIENTNET PARA HAM10000")
    print("="*50)
    
    # Modelo individual
    print("\n1. EfficientNet-B0:")
    model_b0 = create_efficientnet_model(num_classes=7, variant='b0')
    model_summary(model_b0)
    
    print("\n2. EfficientNet-B3:")
    model_b3 = create_efficientnet_model(num_classes=7, variant='b3')
    model_summary(model_b3)
    
    print("\n3. Ensemble (B0 + B3):")
    ensemble = create_ensemble_model(num_classes=7, variants=['b0', 'b3'])
    model_summary(ensemble)