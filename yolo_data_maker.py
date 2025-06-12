import os
import shutil
from pathlib import Path
import re
import glob

def create_yolo_dataset():
    """
    Create YOLO dataset from the existing directory structure and bounding box file.
    """
    
    # Define paths In shop Clothes Retrieval Benchmark
    # source_data_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/img_highres"
    # bbox_file_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/list_bbox_inshop_yolo.txt"
    # output_yolo_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/yolo_format"
    source_data_path = '/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img_highres'
    bbox_file_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/list_bbox_consumer2shop_high_resyolo.txt"
    output_yolo_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/yolo_format"
    
    # Define clothing categories and their class mapping
    topwear_categories = {
        'Jackets_Vests', 'Shirts_Polos', 'Sweaters', 'Sweatshirts_Hoodies', 
        'Tees_Tanks', 'Blouses_Shirts', 'Cardigans', 'Dresses', 
        'Graphic_Tees', 'Jackets_Coats', 'Rompers_Jumpsuits', 'Suiting','Blouse',
        'Coat','Polo_Shirt','T_Shirt','Tank_Top','Summer_Wear','Chiffon','Lace_Shirt','Dress','Lace_Dress',
        'Sleeveless_Dress',
    }
    
    bottomwear_categories = {
        'Denim', 'Pants', 'Leggings', 'Shorts', 'Skirts', 'Suiting','Jeans','Joggers','Skirt',
        'Suspenders_Skirt'
    }
    
    # Class mapping: BottomWear = 0, TopWear = 1
    def get_class_id(category):
        if category in topwear_categories:
            return 2  # TopWear
        elif category in bottomwear_categories:
            return 1  # BottomWear
        else:
            print(f"Warning: Unknown category '{category}', defaulting to TopWear")
            return -1
    
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
    
    # Use glob to find all image files at once
    print("Finding all image files...")
    image_patterns = [
        f"{source_data_path}/**/*.jpg",
        f"{source_data_path}/**/*.jpeg", 
        f"{source_data_path}/**/*.png"
    ]
    
    all_image_files = []
    for pattern in image_patterns:
        all_image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(all_image_files)} total image files")
    
    # Process images and create YOLO format
    processed_count = 0
    skipped_count = 0
    
    for image_path in all_image_files:
        image_file = Path(image_path)
        
        # Extract path components using glob results
        # Path structure: .../GENDER/CATEGORY/ID/IMAGE_NAME
        path_parts = image_file.parts
        
        # Find the indices of gender, category, id
        try:
            img_highres_idx = None
            for i, part in enumerate(path_parts):
                if part == "img_highres":
                    img_highres_idx = i
                    break
            
            if img_highres_idx is None:
                continue
                
            gender = path_parts[img_highres_idx + 1]
            category_name = path_parts[img_highres_idx + 2] 
            id_name = path_parts[img_highres_idx + 3]
            image_name = path_parts[img_highres_idx + 4]
            
            # Skip if not proper structure
            if not id_name.startswith('id_'):
                continue
                
        except IndexError:
            print(f"Warning: Unexpected path structure for {image_path}")
            continue
        
        # Create the original image path as it appears in bbox file
        original_image_path = f"img_highres/{gender}/{category_name}/{id_name}/{image_name}"
        
        if original_image_path in bounding_boxes:
            # Get class ID for this category
            class_id = get_class_id(category_name)
            
            # Create new filename: id_name + image_name
            new_image_name = f"{gender}_{category_name}_{id_name}_{image_name}"
            new_label_name = f"{gender}_{category_name}_{id_name}_{image_file.stem}.txt"
            
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
    Verify the created dataset using glob
    """
    images_dir = Path(output_path) / "images"
    labels_dir = Path(output_path) / "labels"
    
    # Use glob to find files
    image_files = list(glob.glob(str(images_dir / "*.jpg"))) + \
                  list(glob.glob(str(images_dir / "*.jpeg"))) + \
                  list(glob.glob(str(images_dir / "*.png")))
    
    label_files = list(glob.glob(str(labels_dir / "*.txt")))
    
    print(f"\nDataset Verification:")
    print(f"Images: {len(image_files)}")
    print(f"Labels: {len(label_files)}")
    
    # Check for missing pairs
    image_stems = {Path(f).stem for f in image_files}
    label_stems = {Path(f).stem for f in label_files}
    
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
    
    output_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/yolo_format"
    verify_dataset(output_path)
    
    print("\nDataset creation completed!")
