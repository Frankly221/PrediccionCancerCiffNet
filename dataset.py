import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import multiprocessing

class HAM10000Dataset(Dataset):
    """Dataset HAM10000 optimizado para RTX"""
    def __init__(self, csv_file, image_folders, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_folders = image_folders
        self.transform = transform
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['dx'])
        
        print(f"📊 Clases encontradas: {list(self.label_encoder.classes_)}")
        print(f"📈 Distribución de clases:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = sum(self.df['label'] == i)
            print(f"   {class_name}: {count} imágenes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        # Buscar imagen en las carpetas
        image_path = None
        for folder in self.image_folders:
            potential_path = os.path.join(folder, f"{image_id}.jpg")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"No se encontró la imagen: {image_id}.jpg")
        
        # Cargar imagen
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {image_path}: {e}")
            # Imagen por defecto en caso de error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformaciones optimizadas para RTX
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_data_loaders(csv_file, image_folders, batch_size=16, test_size=0.2):
    """Crear DataLoaders optimizados para RTX"""
    
    # Verificar archivos
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_file}")
    
    for folder in image_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"No se encontró la carpeta: {folder}")
    
    print(f"📂 Cargando datos desde: {csv_file}")
    print(f"📁 Carpetas de imágenes: {image_folders}")
    
    # Cargar CSV
    df = pd.read_csv(csv_file)
    print(f"📊 Total de muestras: {len(df)}")
    
    # Verificar columnas necesarias
    required_cols = ['image_id', 'dx']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en CSV")
    
    # Split estratificado
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['dx'], 
        random_state=42
    )
    
    print(f"🔄 División de datos:")
    print(f"   Entrenamiento: {len(train_df)} muestras")
    print(f"   Validación: {len(val_df)} muestras")
    
    # Guardar splits para reproducibilidad
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    # Crear datasets
    train_data = HAM10000Dataset('train_split.csv', image_folders, train_transform)
    val_data = HAM10000Dataset('val_split.csv', image_folders, val_transform)
    
    # Configurar workers para RTX
    num_workers = min(multiprocessing.cpu_count(), 8)  # Máximo 8 workers
    
    print(f"⚙️  Configuración DataLoader:")
    print(f"   Batch size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Pin memory: True (GPU)")
    
    # DataLoaders optimizados para RTX
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,  # Optimización para GPU
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Para estabilidad con AMP
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False
    )
    
    print(f"✅ DataLoaders creados exitosamente!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_data.label_encoder

def verify_dataset(csv_file, image_folders, sample_size=10):
    """Verificar integridad del dataset"""
    print(f"🔍 Verificando dataset...")
    
    df = pd.read_csv(csv_file)
    
    # Verificar muestra aleatoria
    sample_df = df.sample(n=min(sample_size, len(df)))
    
    missing_images = []
    for _, row in sample_df.iterrows():
        image_id = row['image_id']
        found = False
        
        for folder in image_folders:
            image_path = os.path.join(folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                found = True
                break
        
        if not found:
            missing_images.append(image_id)
    
    if missing_images:
        print(f"⚠️  Imágenes faltantes encontradas: {len(missing_images)}")
        for img_id in missing_images[:5]:  # Mostrar solo las primeras 5
            print(f"   - {img_id}.jpg")
        if len(missing_images) > 5:
            print(f"   ... y {len(missing_images) - 5} más")
    else:
        print(f"✅ Todas las imágenes verificadas están presentes")
    
    return len(missing_images) == 0

if __name__ == "__main__":
    # Test del dataset
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = [
        "datasetHam10000/HAM10000_images_part_1", 
        "datasetHam10000/HAM10000_images_part_2"
    ]
    
    print("🧪 Probando dataset HAM10000...")
    
    # Verificar dataset
    if verify_dataset(csv_file, image_folders):
        print("✅ Dataset verificado correctamente")
        
        # Crear DataLoaders de prueba
        try:
            train_loader, val_loader, label_encoder = create_data_loaders(
                csv_file, image_folders, batch_size=4
            )
            
            # Test de carga
            print("🔄 Probando carga de batch...")
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"Batch {batch_idx + 1}:")
                print(f"  Images shape: {images.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels: {labels.tolist()}")
                
                if batch_idx >= 2:  # Solo probar 3 batches
                    break
            
            print("✅ Dataset funcionando correctamente!")
            
        except Exception as e:
            print(f"❌ Error en DataLoaders: {e}")
    else:
        print("❌ Problemas en la verificación del dataset")