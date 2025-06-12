import cv2
import os
from pathlib import Path

# Class names - update this to match your model's classes
CLASSES = ['Handbags', 'Pants', 'Shirts', 'Shoes', 'Sunglasses']
# {'Handbags': 0, 'Pants': 1, 'Shirts': 2, 'Shoes': 3, 'Sunglasses': 4}
# Colors for each class
COLORS = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]

def yolo_to_bbox(yolo_coords, image_width, image_height):
    """
    Convert YOLO format coordinates to bounding box format.
    
    Parameters:
    - yolo_coords: YOLO format [x_center, y_center, width, height] (normalized 0-1)
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    
    Returns:
    - bbox: Bounding box in format [x_1, y_1, x_2, y_2]
    """
    x_center, y_center, width, height = yolo_coords
    
    # Convert normalized coordinates to pixel coordinates
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    width_pixel = width * image_width
    height_pixel = height * image_height
    
    # Calculate corner coordinates
    x_1 = int(x_center_pixel - width_pixel / 2)
    y_1 = int(y_center_pixel - height_pixel / 2)
    x_2 = int(x_center_pixel + width_pixel / 2)
    y_2 = int(y_center_pixel + height_pixel / 2)
    
    # Ensure coordinates are within image bounds
    x_1 = max(0, x_1)
    y_1 = max(0, y_1)
    x_2 = min(image_width, x_2)
    y_2 = min(image_height, y_2)
    
    return [x_1, y_1, x_2, y_2]

def annotate_yolo_on_image(yolo_annotation_path, image_path, output_directory):
    """
    Draw YOLO annotations on an image and save the result.
    
    Parameters:
    - yolo_annotation_path: Path to the YOLO annotation file (.txt)
    - image_path: Path to the input image
    - output_directory: Directory to save the annotated image
    
    Returns:
    - output_image_path: Path to the saved annotated image
    """
    
    # Check if files exist
    if not os.path.exists(yolo_annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {yolo_annotation_path}")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Get image dimensions
    image_height, image_width = image.shape[:2]
    
    # Generate output filename
    input_filename = Path(image_path).stem
    output_filename = f"{input_filename}_annotation.jpg"
    output_image_path = os.path.join(output_directory, output_filename)
    
    # Read YOLO annotations
    annotations = []
    try:
        with open(yolo_annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        parts = line.split()
                        if len(parts) != 5:
                            print(f"Warning: Line {line_num} has {len(parts)} parts, expected 5. Skipping.")
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        annotations.append([class_id, x_center, y_center, width, height])
                    except ValueError as e:
                        print(f"Warning: Could not parse line {line_num}: '{line}'. Error: {e}")
                        continue
    except Exception as e:
        raise Exception(f"Error reading annotation file: {e}")
    
    print(f"Found {len(annotations)} annotations")
    
    # Draw annotations on image
    for annotation in annotations:
        class_id, x_center, y_center, width, height = annotation
        
        # Convert YOLO coordinates to bounding box
        yolo_coords = [x_center, y_center, width, height]
        x1, y1, x2, y2 = yolo_to_bbox(yolo_coords, image_width, image_height)
        
        # Get color for this class
        color = COLORS[class_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if class_id < len(CLASSES):
            label = CLASSES[class_id]
        else:
            label = f"Class_{class_id}"
        
        # Calculate label size
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        top_left = (x1, y1 - label_size[1] - base_line)
        bottom_right = (x1 + label_size[0], y1)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        print(f"Drew {label} at [{x1}, {y1}, {x2}, {y2}]")
    
    # Save the annotated image
    success = cv2.imwrite(output_image_path, image)
    if not success:
        raise Exception(f"Failed to save image to: {output_image_path}")
    
    print(f"Saved annotated image to: {output_image_path}")
    return output_image_path

def annotate_multiple_images(annotation_dir, image_dir, output_directory):
    """
    Annotate multiple images using their corresponding YOLO annotation files.
    
    Parameters:
    - annotation_dir: Directory containing YOLO annotation files (.txt)
    - image_dir: Directory containing images
    - output_directory: Directory to save annotated images
    
    Returns:
    - List of output image paths
    """
    annotation_files = list(Path(annotation_dir).glob("*.txt"))
    output_paths = []
    
    for annotation_file in annotation_files:
        # Find corresponding image file
        base_name = annotation_file.stem
        
        # Try common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_file = None
        
        for ext in image_extensions:
            potential_image = Path(image_dir) / f"{base_name}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break
        
        if image_file:
            try:
                output_path = annotate_yolo_on_image(
                    str(annotation_file), 
                    str(image_file), 
                    output_directory
                )
                output_paths.append(output_path)
                print(f"✓ Processed: {image_file.name}")
            except Exception as e:
                print(f"✗ Error processing {image_file.name}: {e}")
        else:
            print(f"✗ No corresponding image found for: {annotation_file.name}")
    
    return output_paths

if __name__ == "__main__":
    # Example usage

    # read yolo_annotations_file_andPrepare_for_yolo_inference
    
    # Single image annotation
    yolo_dir_annotation_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_in_shop_test/labels"
    image_dir_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_in_shop_test/images"
    output_directory = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_in_shop_test/annotated_images"
    from glob import glob
    yolo_annotations = sorted(glob("/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_in_shop_test/labels/*.txt"))
    images = sorted(glob("/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_in_shop_test/images/*.jpg"))
    for idx, (yolo_annotation, image) in enumerate(zip(yolo_annotations, images)):
        if idx%100 == 0:
            print(f"Processed {idx} images")
        try:
            output_path = annotate_yolo_on_image(yolo_annotation, image, output_directory)
        except Exception as e:
            print(f"Error: {e}")
    
    # # Example for multiple images
    # annotation_dir = "/path/to/annotations"
    # image_dir = "/path/to/images" 
    # output_directory = "/path/to/output"
    # 
    # output_paths = annotate_multiple_images(annotation_dir, image_dir, output_directory)
    # print(f"Processed {len(output_paths)} images")
