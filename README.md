# Shop The Look: Get outfit inspiration with Agentic RAG 

## Milestone 1: Custom Object Detection with YOLO on Fashion Accessories

This section describes an  object detection pipeline using the YOLOv5 architecture. It includes steps like data annotation, model training, evaluation, and inference. YOLOv5 is an object detection model that balances accuracy and speed, making it ideal for real-world applications.

## What is Object Detection?

Object detection is a fundamental task in computer vision where the goal is to not only classify objects in an image but also to **locate them** with bounding boxes. Unlike image classification which gives a single label for an entire image, object detection can identify **multiple objects** and return both:

- The class of each object (e.g., "car", "person")
- The location of the object in the form of bounding box coordinates

This technology helps in real-world applications such as:

- **Autonomous vehicles** (detecting pedestrians, traffic lights)
- **Surveillance systems** (detecting intrusions or abnormal behavior)
- **Retail analytics** (monitoring customer behavior)
- **Healthcare diagnostics** (detecting tumors or anomalies in scans)

<center>
  <img src="https://neurohive.io/wp-content/uploads/2018/11/object-recognition-e1541510005103.png"
       alt="img"
       style="height: 300px; width: 500px; object-fit: contain; margin-bottom: 5px; display: block;">
</center>

## How Object Detection Works

At its core, object detection is a combination of **localization and classification**. Here's how the process generally works:

1. **Input Image**: The raw image is passed to the model.
2. **Feature Extraction**: A convolutional neural network (CNN) is used to extract spatial features.
3. **Region Proposal / Grid Mapping**: The model either proposes regions of interest (in two-stage detectors) or uses fixed grids (in single-stage detectors like YOLO).
4. **Bounding Box Regression**: For each region/grid, the model predicts coordinates for bounding boxes.
5. **Object Classification**: Each box is classified to determine which object it contains.
6. **Post-processing**: Non-Max Suppression (NMS) removes duplicate boxes and retains the most confident detections.

The output is a set of bounding boxes, class labels, and confidence scores.


## Common Object Detection Algorithms

Over time, object detection algorithms have evolved from traditional methods to deep learning-based models. They are broadly categorized as:

### Two-Stage Detectors
These models first generate region proposals and then classify them.
- **R-CNN (2014)**: Extracts features for each region using CNNs.
- **Fast R-CNN**: Improves speed by using a single CNN feature map for the whole image.
- **Faster R-CNN**: Introduces Region Proposal Networks (RPN) for end-to-end learning.

### Single-Stage Detectors
These models predict object locations and class probabilities in one step.
- **YOLO (You Only Look Once)**: Processes the image in one forward pass.
- **SSD (Single Shot MultiBox Detector)**: Predicts multiple boxes at different scales.
- **RetinaNet**: Uses focal loss to address class imbalance in dense detection.


## Why Choose YOLOv5?

YOLOv5 is one of the most advanced and widely adopted object detection frameworks today. Some key reasons to use YOLOv5 include:

- **Speed**: Achieves real-time detection even on edge devices.
- **Accuracy**: Competes with state-of-the-art models like Faster R-CNN and EfficientDet.
- **Flexibility**: Supports transfer learning and custom dataset training with ease.
- **Pre-trained Models**: Comes with ready-to-use weights trained on COCO.
- **Active Development**: Backed by Ultralytics and an active open-source community.
- **PyTorch-Based**: Easy to integrate with other deep learning projects.


## Data Annotation

In this step, we focus on annotating the full-shot image data that has been scraped, preparing it for training a custom YOLOv5 model. Proper data annotation is crucial as it involves labeling the images with the exact locations and categories of the objects we want our model to detect.

<br>
<center>
  <div style="display: flex; justify-content: center; gap: 10px;">
    <img src="utils/images/5.png" alt="da 1" style="height: 600px; width: 30%; object-fit: cover;">
    <img src="utils/images/6.png" alt="da 2" style="height: 600px; width: 30%; object-fit: cover;">
    <img src="utils/images/7.png" alt="da 3" style="height: 600px; width: 30%; object-fit: cover;">
  </div>
</center>
<br>

### Annotating Images in YOLOv5 Supported Format

The annotation process involves drawing bounding boxes around the target objects within the images and assigning the appropriate labels to each bounding box. 

For object detection pipeline, we are interested in detecting specific clothing and accessories, and the labels we are considering are:

- **Topwear**
- **Bottomwear**
- **Footwear**
- **Handbag**
- **EyeWear**

To train a YOLOv5 model, your data must be in the **YOLO annotation format**:

- Each image has a corresponding `.txt` label file with the same filename.
- Each line in the label file represents one object using the format:
  
```
<class_id> <x_center> <y_center> <width> <height>
```

- All values are normalized (i.e., between 0 and 1) relative to image size.
  
<center>
<img src='utils/images/8.png'>
</center>


### Using Label Studio

For data annotation, we used [Label Studio](https://labelstud.io/guide/quick_start). It offers a user-friendly interface for drawing bounding boxes around objects and assigning labels to them, making it an efficient choice for annotating large datasets.

Here’s a **demo video** showing the data annotation process using Label Studio, which provides a visual guide on how to draw bounding boxes and assign labels effectively.


<video width="640" height="360" controls>
  <source src="utils/video/data_annotation.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


[Click to watch the demo video](utils/video/data_annotation.mp4)




## Dataset 

[![In-Shop Dataset | Papers With Code](./static/a60e38d9-01df-40ec-b806-3b0ac98e82c9.png)](https://paperswithcode.com/dataset/in-shop)


## DeepFashion: In-shop Clothes Retrieval Dataset (Customized for Object Detection)

The **DeepFashion: In-shop Clothes Retrieval** dataset is a large-scale benchmark originally developed to support fine-grained image retrieval of fashion items. It focuses on images taken in controlled shop-like environments and features multiple views of the same clothing items, making it well-suited for training deep learning models.

We have **repurposed** this dataset for an **object detection task** by adding bounding box annotations around key fashion components such as **Topwear**, **Bottomwear**, **Shoes**, **Handbags**, and **Sunglasses**.


### Original Dataset Features

- **7,982** unique clothing items
- **52,712** high-quality clothing images
- ~**200,000** image pairs with cross-pose and scale variations
- Annotations include:
  - Bounding boxes (partial)
  - Clothing identity and category
  - Pose types and landmarks
  - Dense pose & parsing masks
  - Attribute labels


### Custom Annotations

To adapt this dataset for object detection, we manually added bounding box annotations for the following fashion parts:

- 👕 **Topwear** 
- 👖 **Bottomwear** 
- 👟 **Footwear** *(annotated with Label Studio)*
- 👜 **Handbags** *(annotated with Label Studio)*
- 🕶️ **EyeWear** *(planned annotations)*

All annotations follow the **YOLO format** and are split into training and validation sets accordingly.


## Training: Object Detection 

### Available YOLOv5 Model Variants

YOLOv5 provides five pre-defined model architectures optimized for different trade-offs between speed and accuracy:

| Model        | Size | Speed (FPS) | Accuracy (mAP) |
|--------------|------|-------------|----------------|
| YOLOv5-nano  | 🟢 Smallest | Fastest     | 🟡 Lower     |
| YOLOv5-small | 🔸 Small    | Fast        | 🟡 Moderate  |
| YOLOv5-medium| ⚫ Medium   | Balanced    | ✅ Good      |
| YOLOv5-large | 🔵 Large    | Balanced    | ✅ Better    |
| YOLOv5-xlarge| 🔴 X-Large  | Slower      | 🔥 Highest   |


### Training the YOLOv5-Large Model

For this project, we selected the **YOLOv5-large** model. It offers a strong balance between inference speed and detection accuracy, making it ideal for identifying fine-grained fashion items like **Handbags, Topwear, Bottomwear, Footwear** and **Sunglasses**.

We followed the official [YOLOv5 training documentation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) to set up and run the training process using **PyTorch**.


### Customizing YOLOv5 for Fashion Object Detection

Although YOLOv5 is pre-trained on 80 COCO classes, we adapt it for **custom object detection** by training on annotated clothing/accessory items.

**Custom Classes for Detection:**
- 👕 **Topwear** 
- 👖 **Bottomwear** 
- 👟 **Footwear** 
- 👜 **Handbags** 
- 🕶️ **EyeWear**

To support this, we configured a custom `.yaml` file to specify dataset paths and class labels.

### YAML Configuration

The `data.yaml` file tells YOLOv5 where to find your data and what classes to detect.

```yaml
# Number of classes
nc: 5

# Names of the classes
names: ['Handbags', 'BottomWear', 'TopWear', 'FootWear', 'EyeWear']

# Paths to the datasets
path: /workspace/Yolo_final                # dataset root dir
train: /workspace/Yolo_final/images/train  # train images (relative to 'path') 450 images
val: /workspace/Yolo_final/images/val      # val images (relative to 'path') 50 images
test:                                        # test images (optional)

# Classes
names:
  0: Handbags
  1: BottomWear
  2: TopWear
  3: FootWear
  4: EyeWear
```

This configuration ensures that YOLOv5 understands the custom dataset labels.

Once the dataset and config are ready, launch training with the following command:

```python
python train.py --img 640 --batch 64 --epochs 40 --weights yolov5l.pt --data /workspace/Yolo_final/pinterest.yaml
```

with following parameters:

- `img`: *Image size*

- `batch`: *Batch size (adjust per GPU memory)*

- `epochs`: *Number of training epochs*

- `data`: *Path to your data.yaml file*

- `weights`: *Pretrained weights to fine-tune from*


----

## Model Performance & Visualizations

To evaluate the performance of our custom YOLOv5-Large fashion detector, we use a combination of standard object detection metrics and training curves provided by YOLOv5.

### Training Curves

Below are the training and validation loss curves over epochs:

- **Box Loss**: Measures bounding box prediction accuracy.
- **Objectness Loss**: Indicates the model's ability to distinguish between object vs background.
- **Classification Loss**: Tracks accuracy in predicting the correct object class.

![Loss Curves](utils/results/results.png)


### Precision–Recall & Confidence Curves

YOLOv5 generates several evaluation plots post-training. These help assess how confidently and accurately the model predicts each class.

#### Precision vs Confidence Curve

- How model precision varies with the confidence threshold.
- Higher curves mean fewer false positives at different thresholds.

![Precision-Confidence](utils/results/P_curve.png)


#### Recall vs Confidence Curve

- Model's ability to detect all relevant objects across varying confidence levels.
- High recall at low confidence suggests the model rarely misses objects.

![Recall-Confidence](utils/results/R_curve.png)


#### Precision–Recall (PR) Curve

- Trade-off between precision and recall for each class.
- Area under the PR curve (AP) indicates model effectiveness. AUC closer to 1 is ideal.

![Precision-Recall Curve](utils/results/PR_curve.png)


#### F1 Score vs Confidence Curve

- Harmonic mean of precision and recall at different confidence thresholds.
- Helps determine the best threshold for optimal balance between precision and recall.

![F1-Confidence](utils/results/F1_curve.png)

### Summary

These visualizations help us:

- Select the best confidence threshold for inference
- Analyze under- or over-fitting (via loss curves)
- Identify potential class imbalance or weak categories
- Tune future training runs for improved performance


> All plots are auto-generated by YOLOv5 and saved under the `runs/train/yolo-fashion-detector/` directory.


### Download the Original Dataset

You can access the official dataset and learn more about it here:  

[DeepFashion: In-shop Clothes Retrieval](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)


### Citation

If you utilize this dataset in your research or projects, please cite the following paper:

```bibtex
@inproceedings{liuLQWTcvpr16DeepFashion,
  author    = {Ziwei Liu and Ping Luo and Shi Qiu and Xiaogang Wang and Xiaoou Tang},
  title     = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2016}
}
```


