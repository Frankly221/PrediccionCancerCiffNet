import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

class HAM10000DatasetPhase2(Dataset):
    """Dataset modificado para CIFF-Net Fase 2 - Devuelve imagen principal + contextuales"""
    def __init__(self, csv_file, image_folders, transform=None, num_context_images=3, phase='train'):
        self.data = pd.read_csv(csv_file)
        self.image_folders = image_folders
        self.transform = transform
        self.num_context_images = num_context_images  # M imágenes contextuales
        self.phase = phase
        
        # Crear mapeo de imágenes
        self.image_mapping = self._create_image_mapping()
        
        # Codificar etiquetas dx
        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['dx'])
        
        # Filtrar imágenes existentes
        existing_images = self.data['image_id'].isin(self.image_mapping.keys())
        self.data = self.data[existing_images].reset_index(drop=True)
        
        # Crear grupos por clase para selección contextual
        self.class_groups = self._create_class_groups()
        
        # Crear grupos por metadatos (edad, sexo, localización) para contexto más rico
        self.metadata_groups = self._create_metadata_groups()
        
        print(f"Dataset Fase 2 cargado: {len(self.data)} imágenes")
        print(f"Imágenes contextuales por muestra: {num_context_images}")
        print(f"Clases: {list(self.label_encoder.classes_)}")
        
    def _create_image_mapping(self):
        image_map = {}
        for folder in self.image_folders:
            if os.path.exists(folder):
                for image_file in os.listdir(folder):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_id = os.path.splitext(image_file)[0]
                        image_map[image_id] = os.path.join(folder, image_file)
        return image_map
    
    def _create_class_groups(self):
        """Agrupar imágenes por clase"""
        class_groups = {}
        for idx, row in self.data.iterrows():
            label = row['label']
            if label not in class_groups:
                class_groups[label] = []
            class_groups[label].append(idx)
        return class_groups
    
    def _create_metadata_groups(self):
        """Agrupar por metadatos para contexto más diverso"""
        metadata_groups = {
            'age_sex': {},
            'localization': {},
            'dx_type': {}
        }
        
        for idx, row in self.data.iterrows():
            # Agrupar por edad y sexo
            age_group = 'young' if row.get('age', 50) < 40 else 'old'
            sex = row.get('sex', 'unknown')
            age_sex_key = f"{age_group}_{sex}"
            
            if age_sex_key not in metadata_groups['age_sex']:
                metadata_groups['age_sex'][age_sex_key] = []
            metadata_groups['age_sex'][age_sex_key].append(idx)
            
            # Agrupar por localización
            loc = row.get('localization', 'unknown')
            if loc not in metadata_groups['localization']:
                metadata_groups['localization'][loc] = []
            metadata_groups['localization'][loc].append(idx)
            
            # Agrupar por tipo de diagnóstico
            dx = row.get('dx', 'unknown')
            if dx not in metadata_groups['dx_type']:
                metadata_groups['dx_type'][dx] = []
            metadata_groups['dx_type'][dx].append(idx)
        
        return metadata_groups
    
    def _select_context_images(self, main_idx, main_label):
        """Seleccionar imágenes contextuales usando múltiples estrategias"""
        context_indices = []
        main_row = self.data.iloc[main_idx]
        
        # Estrategia 1: Misma clase (40% probabilidad)
        if random.random() < 0.4 and len(self.class_groups[main_label]) > 1:
            same_class_candidates = [i for i in self.class_groups[main_label] if i != main_idx]
            if same_class_candidates:
                context_indices.append(random.choice(same_class_candidates))
        
        # Estrategia 2: Misma localización (30% probabilidad)
        if len(context_indices) < self.num_context_images and random.random() < 0.3:
            main_loc = main_row.get('localization', 'unknown')
            if main_loc in self.metadata_groups['localization']:
                loc_candidates = [i for i in self.metadata_groups['localization'][main_loc] if i != main_idx]
                if loc_candidates:
                    context_indices.append(random.choice(loc_candidates))
        
        # Estrategia 3: Similar edad/sexo (20% probabilidad)
        if len(context_indices) < self.num_context_images and random.random() < 0.2:
            age_group = 'young' if main_row.get('age', 50) < 40 else 'old'
            sex = main_row.get('sex', 'unknown')
            age_sex_key = f"{age_group}_{sex}"
            
            if age_sex_key in self.metadata_groups['age_sex']:
                demo_candidates = [i for i in self.metadata_groups['age_sex'][age_sex_key] if i != main_idx]
                if demo_candidates:
                    context_indices.append(random.choice(demo_candidates))
        
        # Estrategia 4: Clases diferentes para contraste (rellenar resto)
        while len(context_indices) < self.num_context_images:
            # Seleccionar de todas las clases excepto la principal
            different_classes = [k for k in self.class_groups.keys() if k != main_label]
            if different_classes:
                diff_class = random.choice(different_classes)
                diff_candidates = self.class_groups[diff_class]
                if diff_candidates:
                    context_indices.append(random.choice(diff_candidates))
            else:
                # Fallback: cualquier imagen diferente
                all_indices = list(range(len(self.data)))
                available = [i for i in all_indices if i != main_idx and i not in context_indices]
                if available:
                    context_indices.append(random.choice(available))
                else:
                    break
        
        # Si no hay suficientes, duplicar algunas
        while len(context_indices) < self.num_context_images:
            if context_indices:
                context_indices.append(random.choice(context_indices))
            else:
                # Último recurso: usar la imagen principal
                context_indices.append(main_idx)
        
        return context_indices[:self.num_context_images]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        main_row = self.data.iloc[idx]
        main_image_id = main_row['image_id']
        main_label = main_row['label']
        
        # Cargar imagen principal
        main_image_path = self.image_mapping[main_image_id]
        main_image = Image.open(main_image_path).convert('RGB')
        
        if self.transform:
            main_image = self.transform(image=np.array(main_image))['image']
        
        # Durante entrenamiento, seleccionar imágenes contextuales
        if self.phase == 'train':
            context_indices = self._select_context_images(idx, main_label)
            
            context_images = []
            context_labels = []
            
            for ctx_idx in context_indices:
                ctx_row = self.data.iloc[ctx_idx]
                ctx_image_id = ctx_row['image_id']
                ctx_label = ctx_row['label']
                
                ctx_image_path = self.image_mapping[ctx_image_id]
                ctx_image = Image.open(ctx_image_path).convert('RGB')
                
                if self.transform:
                    ctx_image = self.transform(image=np.array(ctx_image))['image']
                
                context_images.append(ctx_image)
                context_labels.append(ctx_label)
            
            # Retornar imagen principal + contextuales + metadatos
            return {
                'main_image': main_image,
                'context_images': torch.stack(context_images),  # [M, C, H, W]
                'main_label': torch.tensor(main_label, dtype=torch.long),
                'context_labels': torch.tensor(context_labels, dtype=torch.long),
                'main_idx': idx,
                'context_indices': context_indices
            }
        else:
            # Durante validación, solo imagen principal
            return main_image, torch.tensor(main_label, dtype=torch.long)

# Mantener dataset original para Fase 1
class HAM10000Dataset(Dataset):
    """Dataset original para Fase 1"""
    def __init__(self, csv_file, image_folders, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folders = image_folders
        self.transform = transform
        
        # Crear mapeo de imágenes
        self.image_mapping = self._create_image_mapping()
        
        # Codificar etiquetas dx
        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['dx'])
        
        # Filtrar imágenes existentes
        existing_images = self.data['image_id'].isin(self.image_mapping.keys())
        self.data = self.data[existing_images].reset_index(drop=True)
        
        print(f"Dataset cargado: {len(self.data)} imágenes")
        print(f"Clases: {list(self.label_encoder.classes_)}")
        
    def _create_image_mapping(self):
        image_map = {}
        for folder in self.image_folders:
            if os.path.exists(folder):
                for image_file in os.listdir(folder):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_id = os.path.splitext(image_file)[0]
                        image_map[image_id] = os.path.join(folder, image_file)
        return image_map
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = row['label']
        
        image_path = self.image_mapping[image_id]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        
        return image, torch.tensor(label, dtype=torch.long)

# Transformaciones corregidas para albumentations versión actual
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),  # Cambio: RandomHorizontalFlip -> HorizontalFlip
    A.VerticalFlip(p=0.5),    # Cambio: RandomVerticalFlip -> VerticalFlip
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
    ], p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def create_data_loaders(csv_file, image_folders, batch_size=32, test_size=0.2):
    """DataLoaders para Fase 1"""
    # Dividir datos
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(
        df, test_size=test_size, 
        stratify=df['dx'], 
        random_state=42
    )
    
    # Guardar splits
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    # Crear datasets
    train_data = HAM10000Dataset('train_split.csv', image_folders, train_transform)
    val_data = HAM10000Dataset('val_split.csv', image_folders, val_transform)
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, train_data.label_encoder

def create_data_loaders_phase2(csv_file, image_folders, batch_size=16, test_size=0.2, num_context_images=3):
    """DataLoaders para Fase 2 - Con imágenes contextuales"""
    # Dividir datos
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(
        df, test_size=test_size, 
        stratify=df['dx'], 
        random_state=42
    )
    
    # Guardar splits
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    
    # Crear datasets
    train_data = HAM10000DatasetPhase2(
        'train_split.csv', image_folders, train_transform, 
        num_context_images, phase='train'
    )
    val_data = HAM10000DatasetPhase2(
        'val_split.csv', image_folders, val_transform, 
        num_context_images, phase='val'
    )
    
    # Función de collate personalizada para manejar estructura compleja
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            # Entrenamiento - estructura compleja
            main_images = torch.stack([item['main_image'] for item in batch])
            context_images = torch.stack([item['context_images'] for item in batch])
            main_labels = torch.stack([item['main_label'] for item in batch])
            context_labels = torch.stack([item['context_labels'] for item in batch])
            
            return {
                'main_images': main_images,      # [B, C, H, W]
                'context_images': context_images, # [B, M, C, H, W]
                'main_labels': main_labels,      # [B]
                'context_labels': context_labels # [B, M]
            }
        else:
            # Validación - estructura simple
            images = torch.stack([item[0] for item in batch])
            labels = torch.stack([item[1] for item in batch])
            return images, labels
    
    # Crear DataLoaders con collate personalizado
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, train_data.label_encoder

# Uso
if __name__ == "__main__":
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/HAM10000_images_part_1", "datasetHam10000/HAM10000_images_part_2"]  # Rutas corregidas
    
    print("Probando Dataset Fase 2...")
    train_loader, val_loader, label_encoder = create_data_loaders_phase2(
        csv_file, image_folders, batch_size=4, num_context_images=3
    )
    
    # Probar un batch
    batch = next(iter(train_loader))
    print(f"Main images: {batch['main_images'].shape}")
    print(f"Context images: {batch['context_images'].shape}")
    print(f"Main labels: {batch['main_labels'].shape}")
    print(f"Context labels: {batch['context_labels'].shape}")