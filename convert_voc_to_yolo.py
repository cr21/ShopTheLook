import numpy as np
from multiprocessing import Pool
from typing import List, Tuple, Union
from PIL import Image
import os

def convert_bbox_to_yolo(bbox_data: Tuple[List[float], Tuple[int, int], Tuple[int, int]]) -> List[float]:
    """
    Convert a single bounding box from low resolution to high resolution YOLO format
    
    Args:
        bbox_data: Tuple containing:
            - bbox: List[float] - [x1, y1, x2, y2] coordinates in low resolution
            - low_res_size: Tuple[int, int] - (width, height) of low resolution image
            - high_res_size: Tuple[int, int] - (width, height) of high resolution image
    
    Returns:
        List[float]: Normalized coordinates in YOLO format [x_center, y_center, width, height]
    """
    bbox, (low_width, low_height), (high_width, high_height) = bbox_data
    
    # Calculate scaling factors
    width_scale = high_width / low_width
    height_scale = high_height / low_height
    
    # Scale the coordinates to high resolution
    x1, y1, x2, y2 = bbox
    x1 = x1 * width_scale
    x2 = x2 * width_scale
    y1 = y1 * height_scale
    y2 = y2 * height_scale
    
    # Calculate width and height
    width = x2 - x1
    height = y2 - y1
    
    # Calculate center coordinates
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    
    # Normalize by high resolution dimensions
    x_center_norm = x_center / high_width
    y_center_norm = y_center / high_height
    width_norm = width / high_width
    height_norm = height / high_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]

def get_high_res_path(low_res_path: str) -> str:
    """Convert low resolution path to high resolution path"""
    parts = low_res_path.split('/')
    # Replace 'img' with 'img_highres' and add id prefix to filename
    parts[0] = 'img_highres'
    filename = parts[-1]
    id_prefix = parts[-2]  # Get the ID from parent folder
    parts[-1] = f"{id_prefix}_{filename}"
    return '/'.join(parts)

def convert_bboxes_parallel(bboxes: List[List[float]], 
                          img_size: Tuple[int, int], 
                          num_processes: int = None) -> List[List[float]]:
    """
    Convert multiple bounding boxes from [x1, y1, x2, y2] format to YOLO format in parallel
    
    Args:
        bboxes: List of bounding boxes in [x1, y1, x2, y2] format
        img_size: Tuple of (image_width, image_height)
        num_processes: Number of processes to use. If None, uses all available CPU cores
    
    Returns:
        List of bounding boxes in YOLO format [x_center, y_center, width, height]
    """
    # Prepare data for parallel processing
    bbox_data = [(bbox, img_size) for bbox in bboxes]
    
    # Create process pool and convert boxes in parallel
    with Pool(processes=num_processes) as pool:
        yolo_bboxes = pool.map(convert_bbox_to_yolo, bbox_data)
    
    return yolo_bboxes

# Example usage
if __name__ == "__main__":
    # consumer to shop
    low_res_base_path = '/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img'
    high_res_base_path = '/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img'
    source_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/list_bbox_consumer2shop.txt'
    output_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/Consumer-to-shop Clothes Retrieval Benchmark/list_bbox_consumer2shop_high_resyolo.txt'

    # in shop retrieval
    
    # low_res_base_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark'
    # high_res_base_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark'
    # source_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/list_bbox_inshop.txt'
    # output_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/In-shop Clothes Retrieval Benchmark/list_bbox_inshop_high_res_yolo.txt'

    # Statistics for debugging
    total_processed = 0
    successful = 0
    failed = 0
    
    with open(source_path, 'r') as f:
        total_images = int(f.readline().strip())
        _ = f.readline()  # Skip header
        
        output_lines = []
        output_lines.append(str(total_images) + '\n')
        output_lines.append('image_name clothes_type pose_type x_center y_center width height\n')
        
        for line in f:
            total_processed += 1
            parts = line.strip().split()
            if len(parts) != 7:
                failed += 1
                continue
                
            low_res_image = parts[0]
            clothes_type = parts[1]
            pose_type = parts[2]
            bbox = [float(x) for x in parts[3:7]]  # [x1, y1, x2, y2]
            
            # Get high resolution image path
            high_res_image = get_high_res_path(low_res_image)
            
            try:
                # Get both image dimensions
                with Image.open(os.path.join(low_res_base_path, low_res_image)) as img:
                    low_width, low_height = img.size
                with Image.open(os.path.join(high_res_base_path, high_res_image)) as img:
                    high_width, high_height = img.size
                
                # Convert bbox to YOLO format
                yolo_bbox = convert_bbox_to_yolo((bbox, (low_width, low_height), (high_width, high_height)))
                
                # Validate coordinates
                if any(coord > 1.0 or coord < 0.0 for coord in yolo_bbox):
                    print(f"Warning: Invalid coordinates for {high_res_image}")
                    print(f"Original bbox: {bbox}")
                    print(f"YOLO bbox: {yolo_bbox}")
                    print(f"Low res size: {low_width}x{low_height}")
                    print(f"High res size: {high_width}x{high_height}")
                    failed += 1
                    continue
                
                output_line = f"{high_res_image} {clothes_type} {pose_type} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                output_lines.append(output_line)
                successful += 1
                
            except Exception as e:
                print(f"Error processing {low_res_image} -> {high_res_image}: {str(e)}")
                failed += 1
                continue
    
    # Write output file
    with open(output_path, 'w') as f:
        f.writelines(output_lines)
    
    # Print statistics
    print(f"\nProcessing completed:")
    print(f"Total processed: {total_processed}")
    print(f"Successful conversions: {successful}")
    print(f"Failed conversions: {failed}")
    print(f"Success rate: {(successful/total_processed)*100:.2f}%")
    print(f"\nOutput written to: {output_path}")
