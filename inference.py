import argparse
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel
from model import SegmentationModel
from utils import ensure_dir, save_mask
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    # Load CLIP text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = clip_text_model.to(device)
    clip_text_model.eval()
    
    # Load segmentation model
    seg_model = SegmentationModel(
        text_hidden_dim=clip_text_model.config.hidden_size,
        pretrained_backbone=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    seg_model.load_state_dict(checkpoint['model_state'])
    seg_model = seg_model.to(device)
    seg_model.eval()
    
    return seg_model, clip_text_model, tokenizer

def preprocess_image(image_path, img_size=512):
    """Preprocess image for inference"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    image_np = np.array(image)
    
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size

def predict(model, clip_model, tokenizer, image_path, prompt, device, img_size=512):
    """
    Run inference on a single image
    
    Args:
        model: Segmentation model
        clip_model: CLIP text encoder
        tokenizer: CLIP tokenizer
        image_path: Path to input image
        prompt: Text prompt
        device: torch device
        img_size: Image size for model input
    
    Returns:
        pred_mask: Binary mask (H, W) with values 0-255
        original_size: (width, height) of original image
    """
    # Preprocess image
    image_tensor, original_size = preprocess_image(image_path, img_size)
    image_tensor = image_tensor.to(device)
    
    # Encode prompt
    encoded = tokenizer(
        prompt,
        padding='max_length',
        max_length=77,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # Get text embedding
    with torch.no_grad():
        text_out = clip_model(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None:
            text_emb = text_out.pooler_output
        else:
            text_emb = text_out.last_hidden_state.mean(dim=1)
        
        # Run segmentation
        logits = model(image_tensor, text_emb)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # Post-process
    pred_mask = probs[0, 0]  # Remove batch and channel dims
    
    # Resize to original size
    pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8))
    pred_mask = pred_mask.resize(original_size, Image.BILINEAR)
    pred_mask = np.array(pred_mask)
    
    return pred_mask, original_size

def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--img_size', type=int, default=512, help='Image size for model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary mask')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.output_dir)
    
    print(f"Loading model from {args.checkpoint}...")
    model, clip_model, tokenizer = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Get list of images
    if os.path.isdir(args.input):
        image_files = [
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
    else:
        image_files = [args.input]
    
    print(f"Processing {len(image_files)} images with prompt: '{args.prompt}'")
    
    # Process each image
    for img_path in tqdm(image_files):
        # Run prediction
        pred_mask, original_size = predict(
            model, clip_model, tokenizer, 
            img_path, args.prompt, 
            device, args.img_size
        )
        
        # Apply threshold for binary mask
        binary_mask = (pred_mask >= args.threshold * 255).astype(np.uint8) * 255
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        prompt_clean = args.prompt.replace(" ", "_")
        output_name = f"{base_name}__{prompt_clean}.png"
        output_path = os.path.join(args.output_dir, output_name)
        
        # Save mask
        save_mask(binary_mask, output_path)
    
    print(f"Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()
