import os
import shutil
from pathlib import Path
import re

def create_yolo_dataset():
    """
    Create YOLO dataset from the existing directory structure and bounding box file.
    """
    
    # Define paths
    source_data_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_format/img_highres"
    bbox_file_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_format/list_highres_bbox_yolo_format.txt"
    output_yolo_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_data"
    
    # Define clothing categories and their class mapping
    topwear_categories = {
        'Jackets_Vests', 'Shirts_Polos', 'Sweaters', 'Sweatshirts_Hoodies', 
        'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Dresses', 
        'Graphic_Tees', 'Jackets_Coats', 'Rompers_Jumpsuits', 'Suiting'
    }
    
    bottomwear_categories = {
        'Denim', 'Pants', 'Leggings', 'Shorts', 'Skirts', 'Suiting'
    }
    
    # Class mapping: BottomWear = 0, TopWear = 1
    def get_class_id(category):
        if category in topwear_categories:
            return 1  # TopWear
        elif category in bottomwear_categories:
            return 0  # BottomWear
        else:
            print(f"Warning: Unknown category '{category}', defaulting to TopWear")
            return 1
    
    # Create output directories
    images_dir = Path(output_yolo_path) / "images"
    labels_dir = Path(output_yolo_path) / "labels"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and parse bounding box file
    print("Reading bounding box file...")
    bounding_boxes = {}
    
    try:
        with open(bbox_file_path, 'r') as f:
            lines = f.readlines()
        
        # Skip the first line (number of images)
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 7:
                image_name = parts[0]
                clothes_type = parts[1]
                pose_type = parts[2]
                x_center = float(parts[3])
                y_center = float(parts[4])
                width = float(parts[5])
                height = float(parts[6])
                
                bounding_boxes[image_name] = {
                    'clothes_type': clothes_type,
                    'pose_type': pose_type,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                }
    
    except FileNotFoundError:
        print(f"Error: Bounding box file not found at {bbox_file_path}")
        return
    except Exception as e:
        print(f"Error reading bounding box file: {e}")
        return
    
    print(f"Found {len(bounding_boxes)} bounding box entries")
    
    # Process images and create YOLO format
    processed_count = 0
    skipped_count = 0
    
    for gender in ['MEN', 'WOMEN']:
        gender_path = Path(source_data_path) / gender
        
        if not gender_path.exists():
            print(f"Warning: {gender} directory not found")
            continue
            
        for category_dir in gender_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            class_id = get_class_id(category_name)
            
            print(f"Processing {gender}/{category_name} (class_id: {class_id})")
            
            for id_dir in category_dir.iterdir():
                if not id_dir.is_dir() or not id_dir.name.startswith('id_'):
                    continue
                    
                id_name = id_dir.name
                
                for image_file in id_dir.iterdir():
                    if not image_file.is_file() or not image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        continue
                    
                    # Create the original image path as it appears in bbox file
                    original_image_path = f"img_highres/{gender}/{category_name}/{id_name}/{image_file.name}"
                    
                    if original_image_path in bounding_boxes:
                        # Create new filename: id_name + image_name
                        new_image_name = f"{id_name}_{image_file.name}"
                        new_label_name = f"{id_name}_{image_file.stem}.txt"
                        
                        # Copy image to YOLO images directory
                        dst_image_path = images_dir / new_image_name
                        shutil.copy2(image_file, dst_image_path)
                        
                        # Create YOLO format label file
                        bbox_data = bounding_boxes[original_image_path]
                        label_content = f"{class_id} {bbox_data['x_center']} {bbox_data['y_center']} {bbox_data['width']} {bbox_data['height']}\n"
                        
                        label_file_path = labels_dir / new_label_name
                        with open(label_file_path, 'w') as f:
                            f.write(label_content)
                        
                        processed_count += 1
                        
                        if processed_count % 100 == 0:
                            print(f"Processed {processed_count} images...")
                    else:
                        skipped_count += 1
                        if skipped_count <= 10:  # Only show first 10 warnings
                            print(f"Warning: No bounding box found for {original_image_path}")
    
    # Create dataset configuration files
    create_dataset_config(output_yolo_path, processed_count)
    
    print(f"\nDataset creation completed!")
    print(f"Processed images: {processed_count}")
    print(f"Skipped images (no bbox): {skipped_count}")
    print(f"Output directory: {output_yolo_path}")

def create_dataset_config(output_path, total_images):
    """
    Create dataset configuration files for YOLO training
    """
    
    # Create data.yaml for YOLO training
    data_yaml_content = f"""# YOLO Dataset Configuration
path: {output_path}  # dataset root dir
train: images  # train images (relative to 'path')
val: images    # val images (relative to 'path') 
test:  # test images (optional)

# Classes
nc: 2  # number of classes
names: ['BottomWear', 'TopWear']  # class names

# Additional info
total_images: {total_images}
"""
    
    with open(Path(output_path) / "data.yaml", 'w') as f:
        f.write(data_yaml_content)
    
    # Create classes.txt
    classes_content = "BottomWear\nTopWear\n"
    with open(Path(output_path) / "classes.txt", 'w') as f:
        f.write(classes_content)
    
    # Create README.md with dataset information
    readme_content = f"""# YOLO Fashion Dataset

## Dataset Information
- Total Images: {total_images}
- Classes: 2 (BottomWear, TopWear)
- Format: YOLO

## Class Mapping
- 0: BottomWear (Denim, Pants, Leggings, Shorts, Skirts, Suiting)
- 1: TopWear (Jackets_Vests, Shirts_Polos, Sweaters, Sweatshirts_Hoodies, Tees_Tanks, Blouses_Shirts, Cardigans, Dresses, Graphic_Tees, Jackets_Coats, Rompers_Jumpsuits, Suiting)

## Directory Structure
{Path(output_path).name}/
├── images/           # All images with renamed format: id_XXXXXXXX_imagename.jpg
├── labels/           # Corresponding YOLO format labels: id_XXXXXXXX_imagename.txt
├── data.yaml         # YOLO training configuration
├── classes.txt       # Class names
└── README.md         # This file
```

## File Naming Convention
Images and labels are renamed from the original format:
- Original: `img_highres/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg`
- YOLO format: `id_00000001_02_1_front.jpg` (and corresponding `.txt` label)

## YOLO Label Format
Each label file contains one line per object:
```
class_id x_center y_center width height
```
All coordinates are normalized (0-1 range).
"""
    
    with open(Path(output_path) / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"Created configuration files:")
    print(f"  - data.yaml")
    print(f"  - classes.txt") 
    print(f"  - README.md")

def verify_dataset(output_path):
    """
    Verify the created dataset
    """
    images_dir = Path(output_path) / "images"
    labels_dir = Path(output_path) / "labels"
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"\nDataset Verification:")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    
    # Check for missing pairs
    image_stems = {f.stem for f in image_files}
    label_stems = {f.stem for f in label_files}
    
    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images without labels")
    if missing_images:
        print(f"Warning: {len(missing_images)} labels without images")
    
    if not missing_labels and not missing_images:
        print("✓ All images have corresponding labels")

if __name__ == "__main__":
    print("Starting YOLO dataset creation...")
    create_yolo_dataset()
    
    output_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_data"
    verify_dataset(output_path)
    
    print("\nDataset creation completed!")
