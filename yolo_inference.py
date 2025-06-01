# importing necessary dependencies
import cv2
import time
import sys
import numpy as np
import os
from matplotlib import pyplot as plt

# load model using OpenCV
def build_model(model_path):
  # load custom YOLOv5
  model = cv2.dnn.readNet(model_path)
  # return loaded model
  return model


# performing object detection
def detect(image, net):
  # convert image to blob
  # mean subtraction and scaling
  blob = cv2.dnn.blobFromImage(image, 1/255.0,
                            (INPUT_WIDTH, INPUT_HEIGHT),
                            swapRB=True, crop=False)

  # set the blob as input to the network
  net.setInput(blob)

  # get prediction from the model
  prediction = net.forward()

  return prediction


# pre-processing
def format_yolov5(frame):

  # defining number of rows and columns in image numpy representation
  row, col, _ = frame.shape

  # finding the maximum between row and column
  _max = max(col, row)

  # initializing result matrix with zeros
  result = np.zeros((_max, _max, 3), np.uint8)

  # copying data from frame ndarray to result
  result[0:row, 0:col] = frame

  return result

def convert_yolo_format(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

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

def convert_bbox_to_yolo(left, top, width, height, img_width, img_height):
    """
    Convert bounding box format [left, top, width, height] to YOLO format.
    
    Parameters:
    - left, top, width, height: Bounding box coordinates
    - img_width, img_height: Image dimensions
    
    Returns:
    - YOLO format [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    # Calculate center coordinates
    x_center = left + width / 2
    y_center = top + height / 2
    
    # Normalize coordinates (0-1 range)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm


# post-preprocessing
def wrap_detection(input_image, output_data, original_image_dims):
    class_ids = []
    confidences = []
    boxes = []

    # fetching number of rows in output_data
    rows = output_data.shape[0]

    # fetching width and height of input_image (square/processed image)
    processed_image_width, processed_image_height, _ = input_image.shape
    
    # Original image dimensions
    original_image_height, original_image_width = original_image_dims

    # x-factor for resizing
    x_factor = processed_image_width / INPUT_WIDTH

    # y-factor for resizing
    y_factor = processed_image_height / INPUT_HEIGHT

    # iterate through detections
    for r in range(rows):
        # fetch bounding box co-ordinates
        row = output_data[r]

        # fetch confidence of the detection
        confidence = row[4]

        # filtering out good detections
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # get index of max class score
            _, _, _, max_idx = cv2.minMaxLoc(classes_scores)
            class_id = max_idx[1]

            if (classes_scores[class_id] > 0.25):
                # append confidence of new bounding box to the list
                confidences.append(confidence)
                class_ids.append(class_id)

                # get coordinated center (x and y) and width and height of the bounding box
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()

                # calculate x-coordinate of top-left point of bounding box
                left = int((x-0.5*w) * x_factor)

                # calculate y-coordinate of top-left point of bounding box
                top = int((y-0.5*h) * y_factor)

                # calculate width of bounding box
                width = int(w * x_factor)

                # calculate height of bounding box
                height = int(h * y_factor)

                # create array of coordinates of the bounding box
                box = np.array([left, top, width, height])
                # append new bounding box coordinates to the list
                boxes.append(box)

    # use non-maximum suppression to avoid multiple bounding boxes for the same object
    print(f"total boxes {len(boxes)}")
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    print(f"total boxes indexes {len(indexes)}")
    
    # define new lists to store class ID's, confidences and bounding boxes
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    result_bbox_coords = []
    yolo_lines = []

    # loop through indices
    for i in indexes:
        print(f"i {i}")
        print(f"confidences {confidences[i]}")
        print(f"class_ids {class_ids[i]}")
        print(f"boxes {boxes[i]}")
        print(f"original_image_width {original_image_width}, original_image_height {original_image_height}")
        
        left, top, width, height = boxes[i]
        
        # Convert to YOLO format using ORIGINAL image dimensions
        x_yolo, y_yolo, w_yolo, h_yolo = convert_bbox_to_yolo(left, top, width, height, original_image_width, original_image_height)
        
        # Create YOLO format line for saving to file
        yolo_line = f"{class_ids[i]} {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        print(f"YOLO format: {yolo_line}")
        yolo_lines.append(yolo_line)
        
        # Convert to [x1, y1, x2, y2] format for correct visualization
        x1, y1 = left, top
        x2, y2 = left + width, top + height
        bbox_coords = [x1, y1, x2, y2]
        
        print(f"left {left}, top {top}, width {width}, height {height}")
        print(f"bbox_coords [x1, y1, x2, y2]: {bbox_coords}")
        
        # add detection data to the lists
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
        result_bbox_coords.append(bbox_coords)

    return result_class_ids, result_confidences, result_boxes, result_bbox_coords, yolo_lines


import cv2
from matplotlib import pyplot as plt


def display_object_detection(frame):
    # Display the input and detected images
    plt.figure(figsize=(8, 8))

    # Show the detected image
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('YOLOv5 Detection')
    plt.axis('off')

    plt.show()


# Main function for the custom object detection
def yolo_detect(img_path, net):
    # Read the image
    frame = cv2.imread(img_path)
    # Proceed forward if the image read above is not None
    if frame is not None:
        # Store original image dimensions
        original_image_height, original_image_width = frame.shape[:2]
        original_image_dims = (original_image_height, original_image_width)
        
        # Make a copy of the original image for displaying later
        original_image = frame.copy()

        # Pre-process the input image
        inputImage = format_yolov5(frame)
        # Object detection using our custom model
        outs = detect(inputImage, net)

        # Post-process the detections - pass original dimensions
        class_ids, confidences, boxes, bbox_coords, yolo_lines = wrap_detection(inputImage, outs[0], original_image_dims)

        # Save YOLO format annotations for labeling/training
        print(f"yolo_lines {yolo_lines}")
        print(img_path.split('/')[-1])
        from pathlib import Path
        yolo_output_path = Path(img_path).stem + "_yolo.txt"
        with open(yolo_output_path, "w") as f:
            for yolo_line in yolo_lines:
                f.write(yolo_line + "\n")
        print(f"Saved YOLO annotations to: {yolo_output_path}")

        # Iterate through the detections for drawing annotations on the image
        for (classid, confidence, bbox) in zip(class_ids, confidences, bbox_coords):
            # bbox is now in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = bbox
            
            # Choosing the color for drawing annotations
            color = colors[int(classid) % len(colors)]
            
            # Drawing the bounding box with correct coordinates
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Preparing the label with class name and confidence score
            label = f"{class_list[classid]} ({confidence:.2f})"
            # Calculating label size
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Drawing a filled rectangle above the bounding box for the label
            top_left = (x1, y1 - label_size[1] - base_line)
            bottom_right = (x1 + label_size[0], y1)
            cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)
            # Writing the label on top of the bounding box
            cv2.putText(frame, label, (x1, y1 - base_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the annotated image
        annotated_img_path = 'annotated_' + img_path.split('/')[-1]
        print(f"Saved annotated image to: {annotated_img_path}")
        
        cv2.imwrite(annotated_img_path, frame)

        return annotated_img_path, yolo_output_path
    

if __name__ == "__main__":

    # width of an image
    INPUT_WIDTH = 640

    # height of an image
    INPUT_HEIGHT = 640

    # confidence threshold for object detection
    CONFIDENCE_THRESHOLD = 0.5

    # making an array of all categories

    CATEGORIES = ['Handbags', 'Pants', 'Shirts', 'Shoes', 'Sunglasses']
    # load classes
    class_list = CATEGORIES
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255)]

    img_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/bad59fac-Screenshot_2024-07-27_at_10.55.41AM.png"
    net = build_model("/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/best.onnx")
    yolo_detect(img_path,net)