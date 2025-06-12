# importing necessary dependencies
import cv2
import time
import sys
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, Manager
import threading
from tqdm import tqdm
import psutil
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import onnxruntime as ort

# Thread-safe file writer with aggressive buffering
class ThreadSafeFileWriter:
    def __init__(self, buffer_size=2000):  # Larger buffer
        self.locks = defaultdict(threading.Lock)
        self.buffers = defaultdict(list)
        self.buffer_size = buffer_size
        
    def write_to_file(self, filepath, content):
        with self.locks[filepath]:
            self.buffers[filepath].append(content)
            
            # Flush buffer when it reaches buffer_size
            if len(self.buffers[filepath]) >= self.buffer_size:
                self._flush_buffer(filepath)
    
    def _flush_buffer(self, filepath):
        if self.buffers[filepath]:
            with open(filepath, 'a', buffering=16384) as f:  # Even larger buffer
                f.writelines(self.buffers[filepath])
                f.flush()
            self.buffers[filepath].clear()
    
    def flush_all(self):
        """Flush all buffers - call this at the end"""
        for filepath in list(self.buffers.keys()):
            with self.locks[filepath]:
                self._flush_buffer(filepath)

# System monitoring
class SystemMonitor:
    def __init__(self, log_file="system_monitor.log"):
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = time.time()
        
    def start_monitoring(self, interval=10):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self, interval):
        with open(self.log_file, 'w') as f:
            f.write("Time,CPU%,Memory%,Available_GB\n")
            
        while self.monitoring:
            try:
                elapsed = time.time() - self.start_time
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                available_gb = memory.available / (1024**3)
                
                print(f"[{elapsed:.0f}s] CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Available: {available_gb:.2f}GB")
                
                with open(self.log_file, 'a') as f:
                    f.write(f"{elapsed:.0f},{cpu_percent:.1f},{memory_percent:.1f},{available_gb:.2f}\n")
                
                time.sleep(interval)
            except:
                break

# Global variables for workers
model = None
file_writer = None

def build_model_onnxruntime(model_path):
    """Load YOLO model using ONNXRuntime - supports batch processing"""
    # Set up providers (CPU and GPU if available)
    providers = ['CPUExecutionProvider']
    
    # Try to use GPU if available
    if ort.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')
    
    # Create inference session
    session = ort.InferenceSession(model_path, providers=providers)
    
    print(f"ONNXRuntime model loaded with providers: {session.get_providers()}")
    return session

def preprocess_image_for_batch(frame, target_size=640):
    """Preprocess single image for batch processing"""
    # Resize and pad to square
    h, w = frame.shape[:2]
    
    # Calculate scale and padding
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    # Convert to RGB and normalize
    rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    transposed = np.transpose(normalized, (2, 0, 1))
    
    return transposed, scale, (pad_w, pad_h)

def detect_batch_onnxruntime(images_batch, session):
    """Perform batch detection using ONNXRuntime"""
    # Stack preprocessed images into batch
    batch_input = np.stack(images_batch, axis=0)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: batch_input})
    
    return outputs[0]  # Return predictions

def convert_bbox_to_yolo(left, top, width, height, img_width, img_height):
    """Convert bounding box format to YOLO format"""
    x_center = left + width / 2
    y_center = top + height / 2
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def process_batch_detections(batch_predictions, batch_scales, batch_pads, original_dims_list, batch_size):
    """Process batch predictions and convert to individual image results"""
    batch_results = []
    
    for batch_idx in range(batch_size):
        # Extract predictions for this image
        if len(batch_predictions.shape) == 3:
            # Shape: [batch_size, num_detections, 85] for YOLOv5
            image_predictions = batch_predictions[batch_idx]
        else:
            # Handle different output formats
            detections_per_image = batch_predictions.shape[1] // batch_size
            start_idx = batch_idx * detections_per_image
            end_idx = start_idx + detections_per_image
            image_predictions = batch_predictions[0][start_idx:end_idx]
        
        # Get scaling and padding info for this image
        scale = batch_scales[batch_idx]
        pad_w, pad_h = batch_pads[batch_idx]
        original_h, original_w = original_dims_list[batch_idx]
        
        # Process detections for this image
        class_ids, confidences, boxes, bbox_coords, yolo_lines = process_single_image_detections(
            image_predictions, scale, pad_w, pad_h, original_w, original_h
        )
        
        batch_results.append({
            'class_ids': class_ids,
            'confidences': confidences,
            'boxes': boxes,
            'bbox_coords': bbox_coords,
            'yolo_lines': yolo_lines
        })
    
    return batch_results

def process_single_image_detections(predictions, scale, pad_w, pad_h, original_w, original_h):
    """Process detections for a single image"""
    confidences = []
    class_ids = []
    boxes = []
    
    # Process each detection
    for detection in predictions:
        confidence = detection[4]
        
        if confidence >= CONFIDENCE_THRESHOLD:
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence > 0.25:
                # Get bounding box coordinates (center format from YOLO)
                x_center, y_center, width, height = detection[:4]
                
                # Convert from model coordinates to original image coordinates
                # Remove padding
                x_center = (x_center - pad_w) / scale
                y_center = (y_center - pad_h) / scale
                width = width / scale
                height = height / scale
                
                # Convert to corner format
                left = int(x_center - width / 2)
                top = int(y_center - height / 2)
                box_width = int(width)
                box_height = int(height)
                
                # Clamp to image boundaries
                left = max(0, min(left, original_w))
                top = max(0, min(top, original_h))
                box_width = min(box_width, original_w - left)
                box_height = min(box_height, original_h - top)
                
                confidences.append(float(confidence * class_confidence))
                class_ids.append(int(class_id))
                boxes.append([left, top, box_width, box_height])
    
    if not boxes:
        return [], [], [], [], []
    
    # Apply NMS
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    
    if len(indexes) == 0:
        return [], [], [], [], []
    
    # Process final results
    result_class_ids = []
    result_confidences = []
    result_boxes = []
    result_bbox_coords = []
    yolo_lines = []
    
    for i in indexes.flatten():
        left, top, box_width, box_height = boxes[i]
        
        # YOLO format conversion
        x_yolo, y_yolo, w_yolo, h_yolo = convert_bbox_to_yolo(
            left, top, box_width, box_height, original_w, original_h
        )
        
        yolo_line = f"{class_ids[i]} {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(yolo_line)
        
        # Bounding box coordinates (x1, y1, x2, y2)
        bbox_coords = [left, top, left + box_width, top + box_height]
        
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
        result_bbox_coords.append(bbox_coords)
    
    return result_class_ids, result_confidences, result_boxes, result_bbox_coords, yolo_lines

def init_worker(model_path, categories, input_width, input_height, confidence_threshold):
    """Initialize each worker process"""
    global model, file_writer, CATEGORIES, class_list, INPUT_WIDTH, INPUT_HEIGHT, CONFIDENCE_THRESHOLD
    
    model = build_model_onnxruntime(model_path)
    file_writer = ThreadSafeFileWriter(buffer_size=1000)
    
    CATEGORIES = categories
    class_list = CATEGORIES
    INPUT_WIDTH = input_width
    INPUT_HEIGHT = input_height
    CONFIDENCE_THRESHOLD = confidence_threshold

def process_image_batch_onnxruntime(batch_data):
    """Process a batch of images using ONNXRuntime"""
    img_paths, no_detections_path = batch_data
    global model, file_writer, class_list
    
    batch_size = len(img_paths)
    processed_images = []
    batch_scales = []
    batch_pads = []
    original_dims_list = []
    valid_paths = []
    
    # Load and preprocess all images in the batch
    for img_path in img_paths:
        try:
            frame = cv2.imread(img_path)
            if frame is not None:
                # Store original dimensions
                original_h, original_w = frame.shape[:2]
                original_dims_list.append((original_h, original_w))
                
                # Preprocess for batch
                preprocessed, scale, (pad_w, pad_h) = preprocess_image_for_batch(frame, INPUT_WIDTH)
                processed_images.append(preprocessed)
                batch_scales.append(scale)
                batch_pads.append((pad_w, pad_h))
                valid_paths.append(img_path)
            else:
                file_writer.write_to_file(no_detections_path, f"{img_path} - READ_ERROR\n")
        except Exception as e:
            file_writer.write_to_file(no_detections_path, f"{img_path} - ERROR: {str(e)}\n")
    
    if not processed_images:
        return 0
    
    try:
        # Batch inference with ONNXRuntime
        batch_predictions = detect_batch_onnxruntime(processed_images, model)
        
        # Process batch results
        batch_results = process_batch_detections(
            batch_predictions, batch_scales, batch_pads, original_dims_list, len(processed_images)
        )
        
        total_detections = 0
        
        # Save results for each image in the batch
        for img_path, results in zip(valid_paths, batch_results):
            yolo_lines = results['yolo_lines']
            class_ids = results['class_ids']
            
            if len(yolo_lines) == 0:
                file_writer.write_to_file(no_detections_path, f"{img_path}\n")
            else:
                total_detections += len(yolo_lines)
                print(f"Total detections: {total_detections}")
                
                # Write results efficiently
                yolo_output_path = "inference_results/consumer_to_shop_path_yolo_inference.txt"
                num_objects_file = f"inference_results/consumer_to_shop_path_yolo_inference_detected_{len(yolo_lines)}_object.txt"
                
                for yolo_line in yolo_lines:
                    file_writer.write_to_file(yolo_output_path, f"{img_path} {yolo_line}\n")
                    file_writer.write_to_file(num_objects_file, f"{img_path} {yolo_line}\n")
                
                # Write by class
                for yolo_line, class_id in zip(yolo_lines, class_ids):
                    class_name = class_list[class_id]
                    class_file = f"yolo_inference_for_{class_name}_consumer_to_shop_path.txt"
                    file_writer.write_to_file(class_file, f"{img_path} {yolo_line}\n")
        
        return total_detections
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        for img_path in valid_paths:
            file_writer.write_to_file(no_detections_path, f"{img_path} - BATCH_ERROR: {str(e)}\n")
        return 0

def create_batches(img_path_list, batch_size):
    """Create batches from image list"""
    batches = []
    for i in range(0, len(img_path_list), batch_size):
        batch = img_path_list[i:i + batch_size]
        batches.append(batch)
    return batches

def batch_generator(img_path_list, batch_size):
    """Generator that yields batches from the image path list."""
    for i in range(0, len(img_path_list), batch_size):
        yield img_path_list[i:i + batch_size]


def run_onnxruntime_batch_inference(img_path_list, model_path, batch_size=16, num_processes=None):
    """Run ONNXRuntime batch parallel YOLO inference"""
    
    if num_processes is None:
        num_processes = min(mp.cpu_count() - 1, 6)  # Conservative for batch processing
    
    print(f"Processing {len(img_path_list)} images in batches of {batch_size} using {num_processes} processes...")
    print(f"Using ONNXRuntime for true batch processing")
    
    # Start monitoring
    monitor = SystemMonitor("yolo_onnxruntime_monitor.log")
    monitor.start_monitoring(interval=15)
    
    # Configuration
    categories = ['Handbags', 'BottomWear', 'TopWear', 'FootWear', 'EyeWear']
    no_detections_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/inference_results/no_detections.txt'
    
    # Create batches
    # batches = create_batches(img_path_list, batch_size)
    # print(f"Created {len(batches)} batches")
    # batches = create_batches(img_path_list, batch_size)
    
    # Prepare batch arguments
    batch_gen = batch_generator(img_path_list, batch_size)
    batch_args_gen = ((batch, no_detections_path) for batch in batch_gen)

    #batch_args = [(batch, no_detections_path) for batch in batches]
    #batch_args = [(batch, no_detections_path) for batch in batch_generator(img_path_list, batch_size)]
    

    start_time = time.time()
    total_detections = 0
    processed_images = 0
    
    try:
        # Use ProcessPoolExecutor for batch processing
        with ProcessPoolExecutor(max_workers=num_processes, 
                                 initializer=init_worker,
                                 initargs=(model_path, categories, 640, 640, 0.5)) as executor:
            
            # Submit all batch tasks
            futures = {executor.submit(process_image_batch_onnxruntime, args): len(args[0]) for args in batch_args_gen}
            
            # Process results with progress bar
            num_batches = (len(img_path_list) + batch_size - 1) // batch_size
            completed_batches = 0
            with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
                for future in as_completed(futures):
                    try:
                        batch_detections = future.result()
                        batch_size_actual = futures[future]
                        
                        total_detections += batch_detections
                        processed_images += batch_size_actual
                        completed_batches += 1
                        
                        # Update progress every few batches
                        if completed_batches % 5 == 0:
                            elapsed = time.time() - start_time
                            rate = processed_images / elapsed
                            eta = (len(img_path_list) - processed_images) / rate if rate > 0 else 0
                            pbar.set_postfix({
                                'rate': f'{rate:.1f} img/s',
                                'processed': processed_images,
                                'detections': total_detections,
                                'ETA': f'{eta/3600:.1f}h'
                            })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        pbar.update(1)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted!")
    
    finally:
        monitor.stop_monitoring()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"ONNXRUNTIME BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {len(img_path_list)}")
        print(f"Processed images: {processed_images}")
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"Processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Average speed: {processed_images/total_time:.2f} images/second")
        print(f"Total detections: {total_detections}")
        print(f"Monitor log: yolo_onnxruntime_monitor.log")

if __name__ == "__main__":
    # Configuration
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    CONFIDENCE_THRESHOLD = 0.5
    CATEGORIES = ['Handbags', 'BottomWear', 'TopWear', 'FootWear', 'EyeWear']
    class_list = CATEGORIES
    
    img_path_dir = '/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img_highres/*/*/*/*.jpg'
    img_path_dir = '/Users/chiragtagadiya/Downloads/Annotated_Data/images/train/*.png'
    img_path_list = glob(img_path_dir)
    model_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/best.onnx"
    model_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/best_multi_batch.onnx"
    model_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/models/best_yolom_25epoch.onnx"
    print(f"Found {len(img_path_list)} images to process")
    
    # For testing (remove this line for full processing)
    #img_path_list = img_path_list[:200]  # Test with 200 images first
    
    # Run ONNXRuntime batch parallel inference
    run_onnxruntime_batch_inference(
        img_path_list=img_path_list,
        model_path=model_path,
        batch_size=16,      # Larger batch size possible with ONNXRuntime
        num_processes=1   # Number of parallel processes
    )

#scp -r -i "aws_pem_ec2_yolo.pem"   /Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/data/yolo_data/images/  ubuntu@ec2-98-84-31-221.compute-1.amazonaws.com:/home/ubuntu/data