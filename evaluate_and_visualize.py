"""
Evaluation and visualization script for prompted segmentation
"""
import argparse
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
from model import SegmentationModel
from seg_dataset import PromptedSegDataset, collate_fn, default_augmentations
from torch.utils.data import DataLoader
from tqdm import tqdm

def iou_score(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Compute IoU score"""
    pred = (pred_mask >= threshold).astype(np.uint8)
    gt = (gt_mask >= 0.5).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + eps)

def dice_score(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Compute Dice score"""
    pred = (pred_mask >= threshold).astype(np.uint8)
    gt = (gt_mask >= 0.5).astype(np.uint8)
    intersection = (pred & gt).sum()
    return (2.0 * intersection) / (pred.sum() + gt.sum() + eps) if (pred.sum() + gt.sum()) > 0 else 1.0

def evaluate_model(model, clip_model, dataloader, device):
    """Evaluate model and return detailed metrics"""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get text embeddings
            text_out = clip_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None:
                text_emb = text_out.pooler_output
            else:
                text_emb = text_out.last_hidden_state.mean(dim=1)
            
            # Forward pass
            logits = model(images, text_emb)
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = masks.cpu().numpy()
            
            # Compute metrics for each sample
            for i in range(probs.shape[0]):
                pred = probs[i, 0]
                gt = gts[i, 0]
                
                iou = iou_score(pred, gt)
                dice = dice_score(pred, gt)
                
                results.append({
                    'image_path': batch['image_paths'][i],
                    'prompt': batch['prompts'][i],
                    'iou': iou,
                    'dice': dice
                })
    
    return pd.DataFrame(results)

def visualize_predictions(model, clip_model, tokenizer, csv_path, output_dir, device, num_samples=10):
    """Create visualization of predictions"""
    from seg_dataset import PromptedSegDataset, default_augmentations
    
    # Load dataset
    dataset = PromptedSegDataset(
        csv_path,
        tokenizer,
        augmentations=default_augmentations(img_size=512, is_train=False),
        img_size=512
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Generating visualizations"):
        sample = dataset[idx]
        
        # Prepare inputs
        image = sample['images'].unsqueeze(0).to(device)
        mask_gt = sample['masks']
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            text_out = clip_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None:
                text_emb = text_out.pooler_output
            else:
                text_emb = text_out.last_hidden_state.mean(dim=1)
            
            logits = model(image, text_emb)
            pred_mask = torch.sigmoid(logits).cpu().numpy()[0, 0]
        
        # Denormalize image for visualization
        img_np = sample['images'].permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(mask_gt[0], cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(img_np)
        axes[3].imshow(pred_mask, cmap='Reds', alpha=0.5)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        # Add prompt as suptitle
        plt.suptitle(f'Prompt: "{sample["prompts"]}"', fontsize=12)
        
        # Save
        base_name = os.path.splitext(os.path.basename(sample['image_paths']))[0]
        output_path = os.path.join(output_dir, f'{base_name}_vis.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of visualizations')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    # Load CLIP
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # Load segmentation model
    seg_model = SegmentationModel(
        text_hidden_dim=clip_model.config.hidden_size,
        pretrained_backbone=False
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    seg_model.load_state_dict(checkpoint['model_state'])
    seg_model = seg_model.to(device)
    seg_model.eval()
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint mIoU: {checkpoint.get('miou', 'N/A'):.4f}")
    
    # Create dataset and dataloader
    print("Loading dataset...")
    dataset = PromptedSegDataset(
        args.test_csv,
        tokenizer,
        augmentations=default_augmentations(img_size=512, is_train=False),
        img_size=512
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Evaluate
    print("\nEvaluating model...")
    results_df = evaluate_model(seg_model, clip_model, dataloader, device)
    
    # Print overall metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {len(results_df)}")
    print(f"Mean IoU: {results_df['iou'].mean():.4f} ± {results_df['iou'].std():.4f}")
    print(f"Mean Dice: {results_df['dice'].mean():.4f} ± {results_df['dice'].std():.4f}")
    print(f"Median IoU: {results_df['iou'].median():.4f}")
    print(f"Median Dice: {results_df['dice'].median():.4f}")
    
    # Per-prompt metrics
    print("\n" + "="*60)
    print("PER-PROMPT METRICS")
    print("="*60)
    prompt_metrics = results_df.groupby('prompt').agg({
        'iou': ['mean', 'std', 'count'],
        'dice': ['mean', 'std']
    }).round(4)
    print(prompt_metrics)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test CSV: {args.test_csv}\n")
        f.write(f"Total samples: {len(results_df)}\n\n")
        f.write(f"Mean IoU: {results_df['iou'].mean():.4f} ± {results_df['iou'].std():.4f}\n")
        f.write(f"Mean Dice: {results_df['dice'].mean():.4f} ± {results_df['dice'].std():.4f}\n")
        f.write(f"Median IoU: {results_df['iou'].median():.4f}\n")
        f.write(f"Median Dice: {results_df['dice'].median():.4f}\n\n")
        f.write("Per-prompt metrics:\n")
        f.write(prompt_metrics.to_string())
    print(f"Summary saved to: {summary_path}")
    
    # Create visualizations
    print(f"\nGenerating {args.num_vis} visualizations...")
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    visualize_predictions(
        seg_model, clip_model, tokenizer,
        args.test_csv, vis_dir, device,
        num_samples=args.num_vis
    )
    print(f"Visualizations saved to: {vis_dir}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()
