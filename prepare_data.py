import os
import json
import csv
from pathlib import Path
import shutil
from PIL import Image
import numpy as np

def coco_to_binary_mask(coco_json_path, output_mask_dir, output_image_dir=None, copy_images=True):
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    os.makedirs(output_mask_dir, exist_ok=True)
    if copy_images and output_image_dir:
        os.makedirs(output_image_dir, exist_ok=True)
    
    image_info = {img['id']: img for img in coco_data['images']}
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    results = []
    source_image_dir = os.path.dirname(coco_json_path)
    
    for img_id, img_data in image_info.items():
        img_filename = img_data['file_name']
        img_height = img_data['height']
        img_width = img_data['width']
        
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                if 'segmentation' in ann:
                    from PIL import ImageDraw
                    mask_img = Image.new('L', (img_width, img_height), 0)
                    draw = ImageDraw.Draw(mask_img)
                    
                    for seg in ann['segmentation']:
                        polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                        draw.polygon(polygon, outline=255, fill=255)
                    
                    mask = np.maximum(mask, np.array(mask_img))
        
        mask_filename = os.path.splitext(img_filename)[0] + '.png'
        mask_path = os.path.join(output_mask_dir, mask_filename)
        Image.fromarray(mask).save(mask_path)
        
        if copy_images and output_image_dir:
            src_img = os.path.join(source_image_dir, img_filename)
            dst_img = os.path.join(output_image_dir, img_filename)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
        
        results.append((img_filename, mask_filename))
    
    return results

def create_csv_for_dataset(data_root, dataset_name, prompts, output_csv):
   
    import random
    
    rows = []
    
    for split in ['train', 'valid']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue
        
        coco_json = os.path.join(split_dir, '_annotations.coco.json')
        if not os.path.exists(coco_json):
            print(f"Warning: {coco_json} not found")
            continue
        
        output_images = os.path.join('data', dataset_name, split, 'images')
        output_masks = os.path.join('data', dataset_name, split, 'masks')
        
        print(f"Processing {split} split...")
        results = coco_to_binary_mask(
            coco_json, 
            output_masks, 
            output_images, 
            copy_images=True
        )
        
        # Create CSV rows
        for img_file, mask_file in results:
            img_path = os.path.join(output_images, img_file)
            mask_path = os.path.join(output_masks, mask_file)
            prompt = random.choice(prompts)
            rows.append([img_path, mask_path, prompt])
    
    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'mask_path', 'prompt'])
        writer.writerows(rows)
    
    print(f"Created {output_csv} with {len(rows)} samples")

def main():
    """
    Main function to prepare both datasets
    """
    # Dataset 1: Drywall Joint/Taping
    print("=" * 50)
    print("Processing Dataset 1: Drywall Joint Detection")
    print("=" * 50)
    
    dataset1_prompts = [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment wall joint"
    ]
    
    # Adjust these paths to where you downloaded the Roboflow data
    dataset1_root = "../seam-detect.v1i.coco/" # Your Roboflow download folder
    
    if os.path.exists(dataset1_root):
        for split in ['train', 'valid']:
            split_dir = os.path.join(dataset1_root, split)
            if os.path.exists(split_dir):
                output_csv = f"data/{'train' if split == 'train' else 'val'}_dataset3.csv"
                
                coco_json = os.path.join(split_dir, '_annotations.coco.json')
                output_images = os.path.join('data', 'dataset1', split, 'images')
                output_masks = os.path.join('data', 'dataset1', split, 'masks')
                
                results = coco_to_binary_mask(coco_json, output_masks, output_images, True)
                
                rows = []
                for img_file, mask_file in results:
                    import random
                    img_path = os.path.join(output_images, img_file)
                    mask_path = os.path.join(output_masks, mask_file)
                    prompt = random.choice(dataset1_prompts)
                    rows.append([img_path, mask_path, prompt])
                
                os.makedirs(os.path.dirname(output_csv), exist_ok=True)
                with open(output_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['image_path', 'mask_path', 'prompt'])
                    writer.writerows(rows)
                
                print(f"Created {output_csv} with {len(rows)} samples")
    else:
        print(f"Dataset 1 not found at {dataset1_root}")
    
    # # Dataset 2: Cracks
    # print("\n" + "=" * 50)
    # print("Processing Dataset 2: Crack Detection")
    # print("=" * 50)
    # 
    # dataset2_prompts = [
    #     "segment crack",
    #     "segment wall crack",
    #     "segment surface crack"
    # ]
    # 
    # dataset2_root = "../cracks.v1i.coco"  # Your Roboflow download folder
    # 
    # if os.path.exists(dataset2_root):
    #     for split in ['train', 'valid']:
    #         split_dir = os.path.join(dataset2_root, split)
    #         if os.path.exists(split_dir):
    #             output_csv = f"data/{'train' if split == 'train' else 'val'}_dataset2.csv"
    #             
    #             coco_json = os.path.join(split_dir, '_annotations.coco.json')
    #             output_images = os.path.join('data', 'dataset2', split, 'images')
    #             output_masks = os.path.join('data', 'dataset2', split, 'masks')
    #             
    #             results = coco_to_binary_mask(coco_json, output_masks, output_images, True)
    #             
    #             rows = []
    #             for img_file, mask_file in results:
    #                 import random
    #                 img_path = os.path.join(output_images, img_file)
    #                 mask_path = os.path.join(output_masks, mask_file)
    #                 prompt = random.choice(dataset2_prompts)
    #                 rows.append([img_path, mask_path, prompt])
    #             
    #             os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    #             with open(output_csv, 'w', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow(['image_path', 'mask_path', 'prompt'])
    #                 writer.writerows(rows)
    #             
    #             print(f"Created {output_csv} with {len(rows)} samples")
    # else:
    #     print(f"Dataset 2 not found at {dataset2_root}")
    # 
    # print("\n" + "=" * 50)
    # print("Data preparation complete!")
    # print("=" * 50)
    #
if __name__ == "__main__":
    main()
