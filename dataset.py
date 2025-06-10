import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Definir transformaciones de preprocesamiento
transform = A.Compose([
    A.Resize(224, 224),  # Cambiar el tamaño de la imagen
    A.RandomHorizontalFlip(),  # Aumentar los datos
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización
    ToTensorV2()  # Convertir a tensor
])

# Cargar el dataset
train_data = datasets.ImageFolder('path_to_train_data', transform=transform)
val_data = datasets.ImageFolder('path_to_val_data', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)