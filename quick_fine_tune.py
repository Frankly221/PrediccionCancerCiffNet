import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
import time
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path

# Importar tu modelo
from main import CiffNetADCComplete

class QuickDataset(Dataset):
    """Dataset simple para fine-tuning"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Buscar todas las imágenes
        self.images = []
        self.labels = []
        
        # Mapeo de carpetas a clases (ajusta según tu estructura)
        class_mapping = {
            'melanoma': 0, 'nevus': 1, 'basal_cell_carcinoma': 2,
            'actinic_keratosis': 3, 'benign_keratosis': 4, 
            'dermatofibroma': 5, 'vascular_lesion': 6
        }
        
        for class_name, class_idx in class_mapping.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
        
        print(f"📊 Dataset cargado: {len(self.images)} imágenes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CiffNetFineTuner:
    """Fine-tuner especializado para CiffNet después de correcciones"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Usando device: {self.device}")
        
        # Cargar modelo con correcciones
        self.model = CiffNetADCComplete(num_classes=7, cliff_threshold=0.15)
        
        # Cargar pesos existentes (con manejo de errores)
        self.load_pretrained_weights(model_path)
        
        self.model = self.model.to(self.device)
        
        # ✅ ESTRATEGIA: CONGELAR PHASE 1, ENTRENAR PHASE 2 Y 3
        self.setup_training_strategy()
    
    def load_pretrained_weights(self, model_path):
        """Cargar pesos con manejo inteligente de incompatibilidades"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                pretrained_dict = checkpoint['model_state_dict']
            else:
                pretrained_dict = checkpoint
            
            model_dict = self.model.state_dict()
            
            # ✅ CARGAR SOLO PESOS COMPATIBLES
            compatible_dict = {}
            incompatible_keys = []
            
            for key, value in pretrained_dict.items():
                if key in model_dict:
                    if value.shape == model_dict[key].shape:
                        compatible_dict[key] = value
                    else:
                        incompatible_keys.append(f"{key}: {value.shape} vs {model_dict[key].shape}")
                else:
                    incompatible_keys.append(f"{key}: not found in new model")
            
            # Actualizar con pesos compatibles
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict)
            
            print(f"✅ Pesos cargados:")
            print(f"   Compatible: {len(compatible_dict)}/{len(pretrained_dict)} pesos")
            print(f"   Incompatible: {len(incompatible_keys)} pesos")
            
            if incompatible_keys:
                print("⚠️ Claves incompatibles (serán reinicializadas):")
                for key in incompatible_keys[:5]:  # Mostrar solo los primeros 5
                    print(f"     {key}")
                if len(incompatible_keys) > 5:
                    print(f"     ... y {len(incompatible_keys) - 5} más")
            
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
            print("🔄 Usando inicialización aleatoria")
    
    def setup_training_strategy(self):
        """Configurar estrategia de entrenamiento selectivo"""
        
        # ✅ CONGELAR PHASE 1 (EfficientNet ya está bien entrenado)
        for name, param in self.model.named_parameters():
            if 'phase1' in name and 'backbone' in name:
                param.requires_grad = False
        
        # ✅ ENTRENAR SOLO CAPAS ESPECÍFICAS
        trainable_params = []
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
            else:
                frozen_params += param.numel()
        
        total_trainable = sum(p.numel() for p in trainable_params)
        
        print(f"🎯 Estrategia de entrenamiento:")
        print(f"   Parámetros congelados: {frozen_params:,}")
        print(f"   Parámetros entrenables: {total_trainable:,}")
        print(f"   Ratio entrenamiento: {total_trainable/(total_trainable+frozen_params)*100:.1f}%")
        
        return trainable_params
    
    def create_data_loaders(self, train_dir, val_dir=None, batch_size=8):
        """Crear data loaders para fine-tuning"""
        
        # ✅ TRANSFORMS CONSERVADORES (no data augmentation agresivo)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),  # Mínimo augmentation
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Crear datasets
        train_dataset = QuickDataset(train_dir, transform=train_transform)
        
        if val_dir and os.path.exists(val_dir):
            val_dataset = QuickDataset(val_dir, transform=val_transform)
        else:
            # Split del train dataset
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
        
        # Crear data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader
    
    def fine_tune(self, train_loader, val_loader, epochs=5, lr=1e-5):
        """Proceso de fine-tuning principal"""
        
        print(f"🚀 Iniciando fine-tuning por {epochs} epochs con LR={lr}")
        
        # ✅ OPTIMIZER CONSERVADOR
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params, 
            lr=lr, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # ✅ SCHEDULER SUAVE
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs, 
            eta_min=lr/10
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # TRAINING
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # VALIDATION  
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            # SCHEDULER STEP
            scheduler.step()
            
            # GUARDAR MEJOR MODELO
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = self.model.state_dict().copy()
                print(f"✅ Nuevo mejor modelo: {best_val_acc:.2f}% accuracy")
            
            # LOGGING
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # CARGAR MEJOR MODELO
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Fine-tuning completado!")
            print(f"   Mejor accuracy: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Entrenar una época"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                # ✅ FORWARD CON VALIDACIONES
                outputs = self.model(data)
                probabilities = outputs['phase3']['probabilities']
                
                # ✅ VALIDAR QUE NO HAY NaN
                if torch.any(torch.isnan(probabilities)):
                    print(f"⚠️ NaN detectado en batch {batch_idx}, saltando...")
                    continue
                
                # LOSS
                loss = criterion(probabilities, targets)
                
                # BACKWARD
                loss.backward()
                
                # ✅ GRADIENT CLIPPING CONSERVADOR
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                # MÉTRICAS
                total_loss += loss.item()
                batch_count += 1
                
                pred_classes = torch.argmax(probabilities, dim=1)
                total += targets.size(0)
                correct += (pred_classes == targets).sum().item()
                
                # LOGGING CADA 10 BATCHES
                if batch_idx % 10 == 0:
                    current_acc = 100. * correct / total if total > 0 else 0
                    print(f"  Batch {batch_idx:3d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | "
                          f"Acc: {current_acc:.1f}%")
            
            except Exception as e:
                print(f"❌ Error en batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, val_loader, criterion):
        """Validar una época"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                try:
                    outputs = self.model(data)
                    probabilities = outputs['phase3']['probabilities']
                    
                    if torch.any(torch.isnan(probabilities)):
                        continue
                    
                    loss = criterion(probabilities, targets)
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    pred_classes = torch.argmax(probabilities, dim=1)
                    total += targets.size(0)
                    correct += (pred_classes == targets).sum().item()
                
                except:
                    continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def save_fine_tuned_model(self, save_path):
        """Guardar modelo fine-tuneado"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'num_classes': 7,
                'cliff_threshold': 0.15
            }
        }, save_path)
        print(f"✅ Modelo guardado en: {save_path}")

# ✅ FUNCIÓN PRINCIPAL PARA USAR
def run_quick_fine_tune(
    model_path="results/models/ciffnet_epoch_100.pth",
    train_data_dir="data/train",  # Ajusta a tu estructura
    val_data_dir=None,
    save_path="results/models/ciffnet_fine_tuned.pth",
    epochs=5,
    batch_size=8,
    learning_rate=1e-5
):
    """
    Función principal para ejecutar fine-tuning rápido
    """
    print("🚀 INICIANDO FINE-TUNING RÁPIDO DE CIFFNET")
    print("=" * 60)
    
    # Crear fine-tuner
    fine_tuner = CiffNetFineTuner(model_path)
    
    # Crear data loaders
    train_loader, val_loader = fine_tuner.create_data_loaders(
        train_data_dir, val_data_dir, batch_size
    )
    
    # Fine-tuning
    best_acc = fine_tuner.fine_tune(
        train_loader, val_loader, epochs, learning_rate
    )
    
    # Guardar modelo
    fine_tuner.save_fine_tuned_model(save_path)
    
    print(f"\n🎯 FINE-TUNING COMPLETADO")
    print(f"   Mejor accuracy: {best_acc:.2f}%")
    print(f"   Modelo guardado: {save_path}")
    
    return fine_tuner.model

if __name__ == "__main__":
    # ✅ EJECUTAR FINE-TUNING
    model = run_quick_fine_tune(
        model_path="results/models/ciffnet_epoch_100.pth",
        train_data_dir="data/train",  # CAMBIAR A TU RUTA
        epochs=3,  # Solo 3 epochs para prueba rápida
        batch_size=4,  # Batch pequeño para empezar
        learning_rate=5e-6  # LR muy conservador
    )