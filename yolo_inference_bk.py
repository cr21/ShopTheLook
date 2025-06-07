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

def build_model(model_path):
    """Load YOLO model using OpenCV - single image only"""
    model = cv2.dnn.readNet(model_path)
    return model

def detect_single_optimized(image, net):
    """Optimized single image detection"""
    # Convert image to blob for single image (batch size = 1)
    blob = cv2.dnn.blobFromImage(image, 1/255.0,
                                (INPUT_WIDTH, INPUT_HEIGHT),
                                swapRB=True, crop=False)
    
    # Set input and get prediction
    net.setInput(blob)
    prediction = net.forward()
    
    return prediction

def format_yolov5_fast(frame):
    """Optimized preprocessing"""
    h, w = frame.shape[:2]
    
    # Only resize if needed
    if h == w and h == INPUT_WIDTH:
        return frame
    
    # Find the maximum dimension
    _max = max(w, h)
    
    # Create result array - more efficient
    result = np.zeros((_max, _max, 3), dtype=np.uint8)
    result[0:h, 0:w] = frame
    
    return result

def convert_bbox_to_yolo(left, top, width, height, img_width, img_height):
    """Convert bounding box format to YOLO format"""
    x_center = left + width / 2
    y_center = top + height / 2
    
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return x_center_norm, y_center_norm, width_norm, height_norm

def wrap_detection_optimized(input_image, output_data, original_image_dims):
    """Optimized detection processing"""
    rows = output_data.shape[0]
    
    if rows == 0:
        return [], [], [], [], []
    
    # Pre-allocate arrays for better performance
    confidences = []
    class_ids = []
    boxes = []
    
    # Get dimensions
    processed_image_height, processed_image_width = input_image.shape[:2]
    original_image_height, original_image_width = original_image_dims
    
    # Calculate factors once
    x_factor = processed_image_width / INPUT_WIDTH
    y_factor = processed_image_height / INPUT_HEIGHT
    
    # Vectorized processing where possible
    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            max_score = np.max(classes_scores)
            class_id = np.argmax(classes_scores)
            
            if max_score > 0.25:
                confidences.append(confidence)
                class_ids.append(class_id)
                
                # Bounding box calculations
                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                
                boxes.append([left, top, width, height])
    
    if not boxes:
        return [], [], [], [], []
    
    # NMS
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
        left, top, width, height = boxes[i]
        
        # YOLO format conversion
        x_yolo, y_yolo, w_yolo, h_yolo = convert_bbox_to_yolo(
            left, top, width, height, original_image_width, original_image_height
        )
        
        yolo_line = f"{class_ids[i]} {x_yolo:.6f} {y_yolo:.6f} {w_yolo:.6f} {h_yolo:.6f}"
        yolo_lines.append(yolo_line)
        
        # Bounding box coordinates
        bbox_coords = [left, top, left + width, top + height]
        
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
        result_bbox_coords.append(bbox_coords)
    
    return result_class_ids, result_confidences, result_boxes, result_bbox_coords, yolo_lines

def init_worker(model_path, categories, input_width, input_height, confidence_threshold):
    """Initialize each worker process"""
    global model, file_writer, CATEGORIES, class_list, INPUT_WIDTH, INPUT_HEIGHT, CONFIDENCE_THRESHOLD
    
    model = build_model(model_path)
    file_writer = ThreadSafeFileWriter(buffer_size=1000)  # Larger buffer
    
    CATEGORIES = categories
    class_list = CATEGORIES
    INPUT_WIDTH = input_width
    INPUT_HEIGHT = input_height
    CONFIDENCE_THRESHOLD = confidence_threshold

def process_single_image_ultra_fast(img_path_and_no_det_path):
    """Ultra-optimized single image processing"""
    img_path, no_detections_path = img_path_and_no_det_path
    global model, file_writer, class_list
    
    try:
        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            file_writer.write_to_file(no_detections_path, f"{img_path} - READ_ERROR\n")
            return 0
        
        # Store original dimensions
        original_image_height, original_image_width = frame.shape[:2]
        original_image_dims = (original_image_height, original_image_width)
        
        # Optimized preprocessing
        inputImage = format_yolov5_fast(frame)
        
        # Single image detection
        outs = detect_single_optimized(inputImage, model)
        
        # Optimized post-processing
        class_ids, confidences, boxes, bbox_coords, yolo_lines = wrap_detection_optimized(
            inputImage, outs[0], original_image_dims
        )
        
        if len(yolo_lines) == 0:
            file_writer.write_to_file(no_detections_path, f"{img_path}\n")
            return 0
        
        # Batch file writing for efficiency
        yolo_output_path = "consumer_to_shop_path_yolo_inference.txt"
        num_objects_file = f"consumer_to_shop_path_yolo_inference_detected_{len(yolo_lines)}_object.txt"
        
        # Write all lines for this image at once
        for yolo_line in yolo_lines:
            file_writer.write_to_file(yolo_output_path, f"{img_path} {yolo_line}\n")
            file_writer.write_to_file(num_objects_file, f"{img_path} {yolo_line}\n")
        
        # Write by class
        for yolo_line, class_id in zip(yolo_lines, class_ids):
            class_name = class_list[class_id]
            class_file = f"yolo_inference_for_{class_name}_consumer_to_shop_path.txt"
            file_writer.write_to_file(class_file, f"{img_path} {yolo_line}\n")
        
        return len(yolo_lines)
        
    except Exception as e:
        file_writer.write_to_file(no_detections_path, f"{img_path} - ERROR: {str(e)}\n")
        return 0

def run_ultra_fast_parallel_inference(img_path_list, model_path, num_processes=None):
    """Run ultra-fast parallel YOLO inference"""
    
    if num_processes is None:
        # Use more processes for single image processing
        num_processes = min(mp.cpu_count(), 12)
    
    print(f"Processing {len(img_path_list)} images using {num_processes} processes...")
    
    # Start monitoring
    monitor = SystemMonitor("yolo_ultra_fast_monitor.log")
    monitor.start_monitoring(interval=15)
    
    # Configuration
    categories = ['Handbags', 'Pants', 'Shirts', 'Shoes', 'Sunglasses']
    no_detections_path = '/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/no_detections.txt'
    
    # Prepare arguments
    img_args = [(img_path, no_detections_path) for img_path in img_path_list]
    
    start_time = time.time()
    total_detections = 0
    
    try:
        # Use ProcessPoolExecutor with more workers
        with ProcessPoolExecutor(max_workers=num_processes, 
                                 initializer=init_worker,
                                 initargs=(model_path, categories, 640, 640, 0.5)) as executor:
            
            # Submit all tasks
            futures = [executor.submit(process_single_image_ultra_fast, args) for args in img_args]
            
            # Process results with progress bar
            completed = 0
            with tqdm(total=len(img_args), desc="Processing images", unit="img") as pbar:
                for future in as_completed(futures):
                    try:
                        detections = future.result()
                        total_detections += detections
                        completed += 1
                        
                        # Update progress more frequently
                        if completed % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            eta = (len(img_args) - completed) / rate if rate > 0 else 0
                            pbar.set_postfix({
                                'rate': f'{rate:.1f} img/s',
                                'detections': total_detections,
                                'ETA': f'{eta/3600:.1f}h'
                            })
                        
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        pbar.update(1)
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted!")
    
    finally:
        monitor.stop_monitoring()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"ULTRA-FAST PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total images: {len(img_path_list)}")
        print(f"Processing time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Average speed: {len(img_path_list)/total_time:.2f} images/second")
        print(f"Total detections: {total_detections}")
        print(f"Monitor log: yolo_ultra_fast_monitor.log")

if __name__ == "__main__":
    # Configuration
    INPUT_WIDTH = 640
    INPUT_HEIGHT = 640
    CONFIDENCE_THRESHOLD = 0.5
    CATEGORIES = ['Handbags', 'Pants', 'Shirts', 'Shoes', 'Sunglasses']
    class_list = CATEGORIES
    
    img_path_dir = '/Users/chiragtagadiya/Documents/dataset_shop_the_look/DeepFashion/Consumer-to-shop Clothes Retrieval Benchmark/Consumer-to-shop Clothes Retrieval Benchmark/img/img_highres/*/*/*/*.jpg'
    img_path_list = glob(img_path_dir)
    model_path = "/Users/chiragtagadiya/Downloads/MyProjects/ShopTheLook/best.onnx"
    
    print(f"Found {len(img_path_list)} images to process")
    
    # For testing (remove this line for full processing)
    img_path_list = img_path_list[:200]  # Test with 1k images first
    
    # Run ultra-fast parallel inference
    run_ultra_fast_parallel_inference(
        img_path_list=img_path_list,
        model_path=model_path,
        num_processes=10  # Increase for more parallelism
    )