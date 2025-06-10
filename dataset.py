import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class HAM10000Dataset(Dataset):
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

# Transformaciones específicas para lesiones de piel
train_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomHorizontalFlip(p=0.5),
    A.RandomVerticalFlip(p=0.5),
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
    # Leer y dividir datos
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
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, train_data.label_encoder

# Uso
if __name__ == "__main__":
    csv_file = "datasetHam10000/HAM10000_metadata.csv"
    image_folders = ["datasetHam10000/part_1", "datasetHam10000/part_2"]
    
    train_loader, val_loader, label_encoder = create_data_loaders(csv_file, image_folders)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")