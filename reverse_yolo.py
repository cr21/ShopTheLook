def yolo_to_bbox(yolo_coords, image_width, image_height):
    """
    Convert YOLO format coordinates to bounding box format.
    
    Parameters:
    - yolo_coords: YOLO format [x_center, y_center, width, height] (normalized 0-1)
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    
    Returns:
    - bbox: Bounding box in format [x_1, y_1, x_2, y_2] where:
            x_1, y_1 = upper left corner coordinates
            x_2, y_2 = lower right corner coordinates
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


def yolo_to_bbox_batch(yolo_coords_list, image_width, image_height):
    """
    Convert multiple YOLO format coordinates to bounding box format.
    
    Parameters:
    - yolo_coords_list: List of YOLO format coordinates [[x_center, y_center, width, height], ...]
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    
    Returns:
    - bbox_list: List of bounding boxes in format [[x_1, y_1, x_2, y_2], ...]
    """
    bbox_list = []
    for yolo_coords in yolo_coords_list:
        bbox = yolo_to_bbox(yolo_coords, image_width, image_height)
        bbox_list.append(bbox)
    return bbox_list


def bbox_to_yolo(bbox_coords, image_width, image_height):
    """
    Convert bounding box format to YOLO format coordinates.
    
    Parameters:
    - bbox_coords: Bounding box format [x_1, y_1, x_2, y_2]
    - image_width: Width of the image in pixels
    - image_height: Height of the image in pixels
    
    Returns:
    - yolo_coords: YOLO format [x_center, y_center, width, height] (normalized 0-1)
    """
    x_1, y_1, x_2, y_2 = bbox_coords
    
    # Calculate center coordinates
    x_center = (x_1 + x_2) / 2
    y_center = (y_1 + y_2) / 2
    
    # Calculate width and height
    width = x_2 - x_1
    height = y_2 - y_1
    
    # Normalize coordinates
    x_center_norm = x_center / image_width
    y_center_norm = y_center / image_height
    width_norm = width / image_width
    height_norm = height / image_height
    
    return [x_center_norm, y_center_norm, width_norm, height_norm]


if __name__ == "__main__":
    # Example usage
    
    # Example 1: Single YOLO coordinate conversion
    # yolo_coords = [0.5, 0.5, 0.4, 0.3]  # Center at (0.5, 0.5), width=0.4, height=0.3
    image_width = 654
    image_height = 1302
    
    # bbox = yolo_to_bbox(yolo_coords, image_width, image_height)
    # print(f"YOLO coordinates: {yolo_coords}")
    # print(f"Bounding box coordinates [x_1, y_1, x_2, y_2]: {bbox}")
    
    # Example 2: Multiple YOLO coordinates conversion
    yolo_coords_list = [
        [0.284562, 0.701229, 0.274194, 0.414747],
        [0.292627, 0.334485, 0.302611, 0.347926],
        [0.271505, 0.084101, 0.103687, 0.037634],
        # [0.5452183026805071, 0.08607198748043818, 0.21808732107220286, 0.025039123630672913],
        # [0.5810469339995119, 0.31533646322378717, 0.6137600321603424, 0.32081377151799684],
        # [0.5857202337367734, 0.6924882629107981, 0.5171785042569382, 0.4147104851330203]
    ]


    
    bbox_list = yolo_to_bbox_batch(yolo_coords_list, image_width, image_height)
    print(f"\nMultiple YOLO coordinates: {yolo_coords_list}")
    print(f"Multiple bounding boxes: {bbox_list}")
    
    # # Example 3: Reverse conversion (bbox to YOLO)
    bbox_coords = bbox_list[0]  # Example bbox
    yolo_converted = bbox_to_yolo(bbox_coords, image_width, image_height)
    print(f"\nBounding box coordinates: {bbox_coords}")
    print(f"Converted to YOLO format: {yolo_converted}")
