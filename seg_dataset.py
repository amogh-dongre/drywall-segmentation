import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PromptedSegDataset(Dataset):
    """
    Dataset for text-prompted segmentation
    Loads images, masks, and text prompts from CSV
    """
    def __init__(self, csv_path, tokenizer, augmentations=None, img_size=512):
        """
        Args:
            csv_path: Path to CSV file with columns [image_path, mask_path, prompt]
            tokenizer: HuggingFace tokenizer for text encoding
            augmentations: Albumentations transform pipeline
            img_size: Target image size (square)
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.augmentations = augmentations
        self.img_size = img_size
        
        print(f"Loaded {len(self.df)} samples from {csv_path}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load mask
        mask_path = row['mask_path']
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Normalize mask to [0, 1]
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask has channel dimension
        if len(mask.shape) == 2:
            mask = mask[None, :, :]  # Add channel dimension
        
        # Tokenize prompt
        prompt = row['prompt']
        encoded = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'images': image,
            'masks': mask,
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'prompts': prompt,
            'image_paths': img_path
        }

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    """
    return {
        'images': torch.stack([item['images'] for item in batch]),
        'masks': torch.stack([torch.as_tensor(item['masks'], dtype=torch.float32) for item in batch]),
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'prompts': [item['prompts'] for item in batch],
        'image_paths': [item['image_paths'] for item in batch]
    }

def default_augmentations(img_size=512, is_train=True):
    """
    Default augmentation pipeline using Albumentations
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
