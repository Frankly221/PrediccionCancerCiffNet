import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Importar tu modelo
from main import CiffNetADCComplete

class HAM10000Dataset(Dataset):
    """Dataset para HAM10000 con archivo CSV y carpetas de imágenes"""
    
    def __init__(self, csv_file, image_folders, transform=None, split='train', test_size=0.2, random_state=42):
        self.image_folders = [Path(folder) for folder in image_folders]
        self.transform = transform
        
        print(f"📊 Cargando HAM10000 metadata desde: {csv_file}")
        self.metadata = pd.read_csv(csv_file)
        
        # Mapeo HAM10000 -> CiffNet clases
        self.class_mapping = {
            'mel': 0,     # Melanoma
            'nv': 1,      # Nevus melanocítico  
            'bcc': 2,     # Carcinoma basocelular
            'akiec': 3,   # Queratosis actínica
            'bkl': 4,     # Queratosis seborreica (benigna)
            'df': 5,      # Dermatofibroma
            'vasc': 6     # Lesiones vasculares
        }
        
        self.class_names = [
            'Melanoma', 'Nevus', 'Basal Cell Carcinoma', 
            'Actinic Keratosis', 'Benign Keratosis', 
            'Dermatofibroma', 'Vascular Lesion'
        ]
        
        # Filtrar diagnósticos válidos
        valid_dx = list(self.class_mapping.keys())
        self.metadata = self.metadata[self.metadata['dx'].isin(valid_dx)]
        print(f"✅ Metadata filtrada: {len(self.metadata)} muestras válidas")
        
        # Split train/validation estratificado
        if len(self.metadata) > 0:
            train_df, val_df = train_test_split(
                self.metadata, 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.metadata['dx']
            )
            
            self.data = train_df.reset_index(drop=True) if split == 'train' else val_df.reset_index(drop=True)
        else:
            self.data = pd.DataFrame()
        
        # Validar que las imágenes existen
        self.valid_samples = []
        self._validate_images()
        
        print(f"📊 HAM10000 {split}: {len(self.valid_samples)} imágenes válidas")
        self._print_class_distribution()
    
    def _validate_images(self):
        """Validar que las imágenes existen en las carpetas"""
        missing_count = 0
        found_count = 0
        
        for idx, row in self.data.iterrows():
            image_id = row['image_id']
            dx = row['dx']
            
            # Buscar imagen en las carpetas
            image_path = self._find_image_path(image_id)
            
            if image_path and image_path.exists():
                self.valid_samples.append({
                    'image_path': image_path,
                    'label': self.class_mapping[dx],
                    'class_name': self.class_names[self.class_mapping[dx]],
                    'diagnosis': dx,
                    'image_id': image_id
                })
                found_count += 1
            else:
                missing_count += 1
        
        print(f"✅ Imágenes encontradas: {found_count}")
        if missing_count > 0:
            print(f"⚠️ Imágenes faltantes: {missing_count}")
    
    def _find_image_path(self, image_id):
        """Buscar imagen en las carpetas HAM10000"""
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for folder in self.image_folders:
            if folder.exists():
                for ext in extensions:
                    image_path = folder / f"{image_id}{ext}"
                    if image_path.exists():
                        return image_path
        return None
    
    def _print_class_distribution(self):
        """Mostrar distribución de clases"""
        if len(self.valid_samples) > 0:
            class_counts = {}
            for sample in self.valid_samples:
                class_name = sample['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            print("📊 Distribución de clases:")
            total = len(self.valid_samples)
            for class_name, count in sorted(class_counts.items()):
                percentage = (count / total) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        image_path = sample['image_path']
        label = sample['label']
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
            
        except Exception as e:
            print(f"❌ Error cargando {image_path}: {e}")
            # Imagen en blanco como fallback
            blank_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label

class CiffNetHAMFineTuner:
    """Fine-tuner para CiffNet con HAM10000"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Usando device: {self.device}")
        
        # Cargar modelo
        self.model = CiffNetADCComplete(num_classes=7, cliff_threshold=0.15)
        self.load_pretrained_weights(model_path)
        self.model = self.model.to(self.device)
        self.setup_training_strategy()
    
    def load_pretrained_weights(self, model_path):
        """Cargar pesos existentes"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
            
            model_dict = self.model.state_dict()
            compatible_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            model_dict.update(compatible_dict)
            self.model.load_state_dict(model_dict)
            
            print(f"✅ Pesos cargados: {len(compatible_dict)}/{len(pretrained_dict)} compatibles")
            
        except Exception as e:
            print(f"⚠️ Error cargando pesos: {e}")
    
    def setup_training_strategy(self):
        """Configurar estrategia de entrenamiento"""
        # Congelar Phase 1 backbone
        for name, param in self.model.named_parameters():
            if 'phase1' in name and 'backbone' in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        
        print(f"🎯 Estrategia de entrenamiento:")
        print(f"   Parámetros congelados: {frozen:,}")
        print(f"   Parámetros entrenables: {trainable:,}")
        print(f"   Ratio entrenamiento: {trainable/(trainable+frozen)*100:.1f}%")
    
    def create_data_loaders(self, csv_file, image_folders, batch_size=8, test_size=0.2):
        """Crear data loaders para HAM10000"""
        
        # Transformaciones
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Crear datasets
        train_dataset = HAM10000Dataset(
            csv_file=csv_file,
            image_folders=image_folders,
            transform=train_transform,
            split='train',
            test_size=test_size
        )
        
        val_dataset = HAM10000Dataset(
            csv_file=csv_file,
            image_folders=image_folders,
            transform=val_transform,
            split='val',
            test_size=test_size
        )
        
        # Verificar datos
        if len(train_dataset) == 0:
            raise ValueError("❌ No hay imágenes de entrenamiento")
        if len(val_dataset) == 0:
            raise ValueError("❌ No hay imágenes de validación")
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        print(f"✅ DataLoaders HAM10000 creados:")
        print(f"   Train: {len(train_loader)} batches ({len(train_dataset)} muestras)")
        print(f"   Val: {len(val_loader)} batches ({len(val_dataset)} muestras)")
        
        return train_loader, val_loader
    
    def fine_tune(self, train_loader, val_loader, epochs=5, lr=1e-5):
        """Fine-tuning process"""
        print(f"🚀 Iniciando fine-tuning HAM10000 - {epochs} epochs, LR={lr}")
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/10)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n📊 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            scheduler.step()
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = self.model.state_dict().copy()
                print(f"✅ Nuevo mejor modelo: {best_val_acc:.2f}%")
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n✅ Fine-tuning completado! Mejor accuracy: {best_val_acc:.2f}%")
        
        return best_val_acc
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Entrenar época"""
        self.model.train()
        total_loss, correct, total, batch_count = 0.0, 0, 0, 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                outputs = self.model(data)
                probabilities = outputs['phase3']['probabilities']
                
                if torch.any(torch.isnan(probabilities)):
                    continue
                
                loss = criterion(probabilities, targets)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
                )
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                pred_classes = torch.argmax(probabilities, dim=1)
                total += targets.size(0)
                correct += (pred_classes == targets).sum().item()
                
                if batch_idx % 20 == 0:
                    current_acc = 100. * correct / total if total > 0 else 0
                    print(f"  Batch {batch_idx:3d} | Loss: {loss.item():.4f} | Acc: {current_acc:.1f}%")
            
            except Exception as e:
                print(f"❌ Error batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, val_loader, criterion):
        """Validar época"""
        self.model.eval()
        total_loss, correct, total, batch_count = 0.0, 0, 0, 0
        
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {'num_classes': 7, 'cliff_threshold': 0.15}
        }, save_path)
        print(f"✅ Modelo guardado: {save_path}")

# ✅ FUNCIÓN PRINCIPAL CORREGIDA PARA HAM10000
def run_ham10000_fine_tune(
    model_path="results/models/ciffnet_epoch_100.pth",
    csv_file="datasetHam10000/HAM10000_metadata.csv",
    image_folders=["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"],
    save_path="results/models/ciffnet_ham10000_fine_tuned.pth",
    epochs=3,
    batch_size=4,
    learning_rate=5e-6
):
    """Fine-tuning con dataset HAM10000 completo"""
    
    print("🚀 FINE-TUNING CIFFNET CON HAM10000")
    print("=" * 60)
    
    # Verificar que los archivos existen
    if not os.path.exists(csv_file):
        print(f"❌ CSV no encontrado: {csv_file}")
        return None
    
    for folder in image_folders:
        if not os.path.exists(folder):
            print(f"❌ Carpeta no encontrada: {folder}")
            return None
    
    print(f"✅ Archivos HAM10000 verificados")
    
    try:
        # Crear fine-tuner
        fine_tuner = CiffNetHAMFineTuner(model_path)
        
        # Crear data loaders
        train_loader, val_loader = fine_tuner.create_data_loaders(
            csv_file, image_folders, batch_size
        )
        
        # Fine-tuning
        best_acc = fine_tuner.fine_tune(train_loader, val_loader, epochs, learning_rate)
        
        # Guardar modelo
        fine_tuner.save_fine_tuned_model(save_path)
        
        print(f"\n🎯 FINE-TUNING HAM10000 COMPLETADO")
        print(f"   Mejor accuracy: {best_acc:.2f}%")
        print(f"   Modelo guardado: {save_path}")
        
        return fine_tuner.model
        
    except Exception as e:
        print(f"❌ Error durante fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # ✅ EJECUTAR FINE-TUNING CON HAM10000
    model = run_ham10000_fine_tune(
        model_path="results/models/ciffnet_epoch_100.pth",
        csv_file="datasetHam10000/HAM10000_metadata.csv",
        image_folders=[
            "datasetHam10000/HAM10000_images_part_1", 
            "datasetHam10000/HAM10000_images_part_2"
        ],
        epochs=3,
        batch_size=4,
        learning_rate=5e-6
    )
    
    if model:
        print("✅ ¡Fine-tuning exitoso! El modelo está listo para usar.")
    else:
        print("❌ Fine-tuning falló. Revisa los errores arriba.")