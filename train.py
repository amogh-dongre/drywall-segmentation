import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from seg_dataset import PromptedSegDataset, collate_fn, default_augmentations
from model import SegmentationModel
from utils import set_seed, ensure_dir, save_mask, timeit
import numpy as np
from tqdm import tqdm
import torch.optim as optim

def dice_loss(pred, target, eps=1e-6):
    """Dice loss for segmentation"""
    p = torch.sigmoid(pred)
    p = p.view(p.shape[0], -1)
    t = target.view(target.shape[0], -1)
    intersect = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    loss = 1 - ((2 * intersect + eps) / (denom + eps))
    return loss.mean()

def iou_score(pred_mask, gt_mask, threshold=0.5, eps=1e-6):
    """Compute IoU score"""
    pred = (pred_mask >= threshold).astype(np.uint8)
    gt = (gt_mask >= 0.5).astype(np.uint8)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + eps)

def compute_metrics_batch(preds, gts):
    """Compute metrics for a batch"""
    ious = []
    dices = []
    for p, g in zip(preds, gts):
        # IoU
        i = iou_score(p, g)
        ious.append(i)
        
        # Dice
        pred_bin = (p >= 0.5).astype(np.uint8)
        gt_bin = g.astype(np.uint8)
        intersection = (pred_bin & gt_bin).sum()
        dice = (2.0 * intersection) / (pred_bin.sum() + gt_bin.sum() + 1e-6) if (pred_bin.sum() + gt_bin.sum()) > 0 else 1.0
        dices.append(dice)
    
    return {'mIoU': float(np.mean(ious)), 'mDice': float(np.mean(dices))}

def train_one_epoch(model, clip_text_model, dataloader, opt, device, loss_weights):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in pbar:
        images = batch['images'].to(device)
        masks = batch['masks'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get text embeddings
        with torch.no_grad():
            text_out = clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None:
                text_emb = text_out.pooler_output
            else:
                text_emb = text_out.last_hidden_state.mean(dim=1)
            text_emb = text_emb.detach()
        
        # Forward pass
        logits = model(images, text_emb)
        
        # Compute loss
        bce = nn.BCEWithLogitsLoss()(logits, masks)
        dloss = dice_loss(logits, masks)
        loss = loss_weights['bce'] * bce + loss_weights['dice'] * dloss
        
        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * images.shape[0]
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader.dataset)

@timeit
def evaluate(model, clip_text_model, dataloader, device, out_dir=None):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_gts = []
    
    if out_dir:
        ensure_dir(out_dir)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for i, batch in enumerate(pbar):
            images = batch['images'].to(device)
            masks = batch['masks'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get text embeddings
            text_out = clip_text_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None:
                text_emb = text_out.pooler_output
            else:
                text_emb = text_out.last_hidden_state.mean(dim=1)
            
            # Forward pass
            logits = model(images, text_emb)
            probs = torch.sigmoid(logits).cpu().numpy()
            gts = masks.cpu().numpy()
            
            # Squeeze channel dimension
            probs = probs.squeeze(1)
            gts = gts.squeeze(1)
            
            all_preds.append(probs)
            all_gts.append(gts)
            
            # Save some predictions
            if out_dir and i < 10:
                for bi in range(probs.shape[0]):
                    pred = (probs[bi] * 255).astype(np.uint8)
                    imgpath = batch['image_paths'][bi]
                    base = os.path.splitext(os.path.basename(imgpath))[0]
                    prompt = batch['prompts'][bi].replace(" ", "_")
                    fname = f"{base}__{prompt}.png"
                    save_mask(pred, os.path.join(out_dir, fname))
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    metrics = compute_metrics_batch(all_preds, all_gts)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()
    
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    
    # Initialize CLIP
    print("Loading CLIP model...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Freeze CLIP
    for p in clip_text_model.parameters():
        p.requires_grad = False
    
    # Create datasets
    print("Loading datasets...")
    train_ds = PromptedSegDataset(
        args.train_csv, 
        tokenizer, 
        augmentations=default_augmentations(args.img_size, is_train=True),
        img_size=args.img_size
    )
    val_ds = PromptedSegDataset(
        args.val_csv, 
        tokenizer, 
        augmentations=default_augmentations(args.img_size, is_train=False),
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4,
        pin_memory=True
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    seg_model = SegmentationModel(
        text_hidden_dim=clip_text_model.config.hidden_size,
        pretrained_backbone=True
    )
    seg_model = seg_model.to(device)
    clip_text_model = clip_text_model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in seg_model.parameters())
    trainable_params = sum(p.numel() for p in seg_model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    opt = optim.AdamW(
        seg_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # Training loop
    best_miou = -1.0
    loss_weights = {'bce': 1.0, 'dice': 1.0}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('='*60)
        
        # Train
        train_loss = train_one_epoch(
            seg_model, clip_text_model, train_loader, opt, device, loss_weights
        )
        
        # Evaluate
        eval_metrics, eval_time = evaluate(
            seg_model, 
            clip_text_model, 
            val_loader, 
            device, 
            out_dir=os.path.join(args.out_dir, "val_preds")
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val mIoU: {eval_metrics['mIoU']:.4f}")
        print(f"Val mDice: {eval_metrics['mDice']:.4f}")
        print(f"Eval Time: {eval_time:.2f}s")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state': seg_model.state_dict(),
            'opt_state': opt.state_dict(),
            'miou': eval_metrics['mIoU'],
            'mdice': eval_metrics['mDice']
        }, ckpt_path)
        
        # Save best model
        if eval_metrics['mIoU'] > best_miou:
            best_miou = eval_metrics['mIoU']
            best_path = os.path.join(args.out_dir, "best.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state': seg_model.state_dict(),
                'opt_state': opt.state_dict(),
                'miou': eval_metrics['mIoU'],
                'mdice': eval_metrics['mDice']
            }, best_path)
            print(f"✓ Saved best model (mIoU: {best_miou:.4f})")
    
    print(f"\n{'='*60}")
    print("Training finished!")
    print(f"Best mIoU: {best_miou:.4f}")
    print('='*60)

if __name__ == "__main__":
    main()
