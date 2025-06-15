import torch
from torch.utils.data import DataLoader
from dataset_improved import *

def create_aggressive_rtx3070_data_loaders(csv_file, image_folders, batch_size=32, test_size=0.2):
    """DataLoaders EXTREMOS para RTX 3070 + 32GB RAM"""
    
    print(f"📂 Cargando datos RTX 3070 AGGRESSIVE desde: {csv_file}")
    
    # Usar clase base pero con parámetros extremos
    train_data = HAM10000Dataset(csv_file, image_folders, transform=get_train_transforms(), is_train=True)
    
    # Split
    train_size = int((1 - test_size) * len(train_data))
    val_size = len(train_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # CONFIGURACIÓN EXTREMA RTX 3070
    num_workers = 20        # ⬆️ EXTREMO 32GB RAM
    pin_memory = True
    prefetch_factor = 8     # ⬆️ EXTREMO throughput
    persistent_workers = True
    
    print(f"⚡ Configuración RTX 3070 AGGRESSIVE:")
    print(f"   Batch size: {batch_size} (EXTREMO)")
    print(f"   Workers: {num_workers} (EXTREMO)")
    print(f"   Prefetch factor: {prefetch_factor} (EXTREMO)")
    print(f"   Memory format: channels_last")
    print(f"   Target GPU: 95%+")
    print(f"   Target VRAM: 7.5GB+")
    
    # DataLoaders EXTREMOS
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        drop_last=True,
        multiprocessing_context='spawn',
        timeout=180,  # ⬆️ Más tiempo para batches grandes
        pin_memory_device='cuda'  # ⬆️ Pin directamente a CUDA
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context='spawn',
        timeout=180,
        pin_memory_device='cuda'
    )
    
    print(f"🚀 DataLoaders RTX 3070 AGGRESSIVE creados!")
    print(f"   Esperado GPU: 95%+")
    print(f"   Esperado VRAM: 7-7.5GB")
    
    return train_loader, val_loader, train_data.label_encoder, train_data.get_class_weights_tensor()