import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torchvision.transforms as transforms
import multiprocessing
from collections import Counter
import random

class AdvancedAugmentation:
    """Augmentaciones avanzadas específicas para dermatología"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            # Aplicar una augmentación aleatoria
            aug_type = random.choice([
                'hair_simulation', 'microscope_artifact', 
                'color_variation', 'texture_enhancement'
            ])
            
            if aug_type == 'hair_simulation':
                return self.add_hair_artifact(img)
            elif aug_type == 'microscope_artifact':
                return self.add_microscope_artifact(img)
            elif aug_type == 'color_variation':
                return self.enhance_color_variation(img)
            elif aug_type == 'texture_enhancement':
                return self.enhance_texture(img)
        
        return img
    
    def add_hair_artifact(self, img):
        """Simular pelos en la imagen"""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Agregar 1-3 "pelos"
        num_hairs = random.randint(1, 3)
        for _ in range(num_hairs):
            # Coordenadas aleatorias
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = x1 + random.randint(-50, 50)
            y2 = y1 + random.randint(-50, 50)
            
            # Color oscuro aleatorio
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))
        
        return img
    
    def add_microscope_artifact(self, img):
        """Simular artefactos de microscopio"""
        # Agregar ruido sutil
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Agregar blur sutil en bordes
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        
        return img
    
    def enhance_color_variation(self, img):
        """Variaciones de color médicamente relevantes"""
        # Simular diferentes condiciones de iluminación
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # Variación de saturación
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
        return img
    
    def enhance_texture(self, img):
        """Realzar texturas médicamente importantes"""
        # Sharpening sutil
        if random.random() < 0.4:
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return img

class BalancedHAM10000Dataset(Dataset):
    """Dataset HAM10000 con balanceado de clases"""
    def __init__(self, csv_file, image_folders, transform=None, oversample_minorities=True):
        self.df = pd.read_csv(csv_file)
        self.image_folders = image_folders
        self.transform = transform
        self.oversample_minorities = oversample_minorities
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['dx'])
        
        # Análisis de distribución
        self.class_counts = Counter(self.df['label'])
        self.class_weights = self._compute_class_weights()
        
        # Sobremuestreo si está habilitado
        if oversample_minorities:
            self.df = self._oversample_minorities()
        
        print(f"📊 Clases encontradas: {list(self.label_encoder.classes_)}")
        self._print_distribution()
    
    def _compute_class_weights(self):
        """Calcular pesos por clase"""
        labels = self.df['label'].values
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        weight_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            weight_dict[class_name] = class_weights[i]
        
        return weight_dict
    
    def _oversample_minorities(self):
        """Sobremuestrear clases minoritarias"""
        # Encontrar la clase mayoritaria
        max_count = max(self.class_counts.values())
        
        # Definir targets de muestreo (no tan agresivo)
        target_counts = {}
        for label, count in self.class_counts.items():
            if count < max_count * 0.2:  # Clases muy pequeñas
                target_counts[label] = min(int(max_count * 0.3), count * 3)
            elif count < max_count * 0.5:  # Clases medianas
                target_counts[label] = min(int(max_count * 0.6), count * 2)
            else:  # Clases grandes
                target_counts[label] = count
        
        # Crear nuevo dataframe balanceado
        balanced_dfs = []
        
        for label, target_count in target_counts.items():
            label_df = self.df[self.df['label'] == label].copy()
            current_count = len(label_df)
            
            if current_count < target_count:
                # Sobremuestrear
                additional_needed = target_count - current_count
                additional_samples = label_df.sample(
                    n=additional_needed, 
                    replace=True, 
                    random_state=42
                )
                balanced_dfs.append(pd.concat([label_df, additional_samples]))
            else:
                balanced_dfs.append(label_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
    
    def _print_distribution(self):
        """Imprimir distribución de clases"""
        print(f"📈 Distribución después del balanceo:")
        current_counts = Counter(self.df['label'])
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = current_counts[i]
            weight = self.class_weights[class_name]
            print(f"   {class_name}: {count} imágenes (peso: {weight:.3f})")
    
    def get_class_weights_tensor(self):
        """Obtener pesos como tensor para loss function"""
        weights = []
        for class_name in self.label_encoder.classes_:
            weights.append(self.class_weights[class_name])
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        # Buscar imagen
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
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transformaciones mejoradas
train_transform_improved = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    AdvancedAugmentation(p=0.3),  # Augmentación médica específica
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),  # Más rotación
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1), 
        scale=(0.8, 1.2),  # Más variación de escala
        shear=10
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))  # Más agresivo
])

val_transform_improved = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_improved_data_loaders(csv_file, image_folders, batch_size=24, test_size=0.2):
    """Crear DataLoaders MAX GPU para RTX 3070 + 32GB RAM"""
    
    print(f"📂 Cargando datos MAX GPU desde: {csv_file}")
    
    # Verificar archivos
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"No se encontró: {csv_file}")
    
    for folder in image_folders:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"No se encontró: {folder}")
    
    # Split estratificado
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['dx'], random_state=42
    )
    
    train_df.to_csv('train_split_improved.csv', index=False)
    val_df.to_csv('val_split_improved.csv', index=False)
    
    # Crear datasets con balanceado
    train_data = BalancedHAM10000Dataset(
        'train_split_improved.csv', 
        image_folders, 
        train_transform_improved,
        oversample_minorities=True
    )
    
    val_data = BalancedHAM10000Dataset(
        'val_split_improved.csv', 
        image_folders, 
        val_transform_improved,
        oversample_minorities=False
    )
    
    # CONFIGURACIÓN MAX GPU
    num_workers = 16        # ⬆️ MÁXIMO para 32GB RAM
    pin_memory = True
    prefetch_factor = 6     # ⬆️ MÁXIMO para throughput
    persistent_workers = True
    
    print(f"⚙️  Configuración MAX GPU:")
    print(f"   Batch size: {batch_size} (MAX RTX 3070)")
    print(f"   Workers: {num_workers} (MAX 32GB RAM)")
    print(f"   Prefetch factor: {prefetch_factor}")
    print(f"   Target GPU: 95%+")
    print(f"   Target VRAM: 7.5GB+")
    
    # DataLoaders MAX GPU
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        multiprocessing_context='spawn',
        timeout=120
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context='spawn',
        timeout=120
    )
    
    print(f"🚀 DataLoaders MAX GPU creados!")
    
    return train_loader, val_loader, train_data.label_encoder, train_data.get_class_weights_tensor()