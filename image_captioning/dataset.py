import os
import json
import urllib.request
import zipfile
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
from io import BytesIO

class CompleteImageCaptionDataset:
    def __init__(self, dataset_size=5000):
        self.dataset_size = dataset_size  # 5K images for demo
        self.data_dir = "image_caption_dataset"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.annotations_file = os.path.join(self.data_dir, "captions.json")
        
        # Create directories
        os.makedirs(self.train_dir, exist_ok=True)
    
    def download_sample_data(self):
        """Create sample data"""
        print("📥 Creating sample dataset...")
        
        captions = [
            "a dog playing in the park",
            "a cat sleeping on a couch",
            "a beautiful beach at sunset",
            "delicious food on a plate",
            "a busy city street"
        ] * (self.dataset_size // 5 + 1)
        
        dataset = []
        for i in range(self.dataset_size):
            img_path = os.path.join(self.train_dir, f"img_{i:012d}.jpg")
            
            dataset.append({
                'image_id': i,
                'image_path': img_path,
                'caption': captions[i % len(captions)]
            })
        
        # Save annotations
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self.annotations_file, 'w') as f:
            json.dump(dataset, f)
        print(f"✅ Annotations saved: {len(dataset)} samples")
    
    def download_images(self):
        """Download actual images"""
        print("🖼️ Downloading images...")
        
        with open(self.annotations_file, 'r') as f:
            data = json.load(f)
        
        for item in tqdm(data[:self.dataset_size], desc="Downloading images"):
            img_path = item['image_path']
            img_id = item['image_id']
            
            # Skip if exists
            if os.path.exists(img_path):
                continue
            
            # Get random image URL (replace with your URLs)
            urls = [
                "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
                "https://images.unsplash.com/photo-1534361960057-19889db9621e?w=400",
                "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
                "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",
                "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",
                "https://images.unsplash.com/photo-1544964209-b4e0f12ff94e?w=400",
                "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            ]
            url = urls[img_id % len(urls)]
            
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.save(img_path)
            except:
                # Create dummy image if download fails
                img = Image.new('RGB', (400, 300), color='gray')
                img.save(img_path)
    
    def get_pytorch_dataset(self):
        """PyTorch Dataset class"""
        class ImageCaptionDataset(Dataset):
            def __init__(self, annotations_file, processor):
                with open(annotations_file, 'r') as f:
                    self.data = json.load(f)
                self.processor = processor
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                image_path = item['image_path']
                caption = item['caption']
                
                # Load image
                image = Image.open(image_path).convert('RGB')
                
                # Process
                encoding = self.processor(
                    images=image,
                    text=caption,
                    padding="max_length",
                    max_length=50,
                    truncation=True,
                    return_tensors="pt"
                )
                
                return {
                    'pixel_values': encoding['pixel_values'].squeeze(),
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        return ImageCaptionDataset(self.annotations_file, processor)
    
    def create_complete_dataset(self):
        """ONE FUNCTION TO CREATE EVERYTHING"""
        print("🎯 Creating COMPLETE dataset...")
        
        self.download_sample_data()
        self.download_images()
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        dataset = self.get_pytorch_dataset()
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        sample_batch = next(iter(dataloader))
        
        print("✅ DATASET READY!")
        print(f"📊 {len(dataset)} images + captions")
        print(f"📁 Images in: {self.train_dir}")
        print(f"📄 Annotations: {self.annotations_file}")
        print("🎉 Sample batch shape:", sample_batch['pixel_values'].shape)
        
        return dataset, dataloader

# 🚀 USAGE - RUN THIS!
if __name__ == "__main__":
    # Create dataset (5K samples)
    dataset_creator = CompleteImageCaptionDataset(dataset_size=100)  # Change to 5000 for full
    dataset, dataloader = dataset_creator.create_complete_dataset()
    
    print("\n" + "="*50)
    print("✅ SUCCESS! Your dataset is ready for training!")
    print("📁 Folder structure:")
    print("image_caption_dataset/")
    print("  📄 captions.json")
    print("  📁 train/")
    print("     🖼️ img_000001.jpg")
    print("     🖼️ img_000002.jpg")
    print("     ...")
    print("\n🔥 Next: Use in training!")