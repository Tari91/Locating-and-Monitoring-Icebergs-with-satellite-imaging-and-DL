🧊 Iceberg Detection and Monitoring using Synthetic Satellite Data + Deep Learning
Overview

This project demonstrates how to locate and monitor icebergs using synthetic satellite imagery generated programmatically and analyzed using a U-Net deep learning model (TensorFlow/Keras).

It creates realistic ocean and iceberg scenes, trains a segmentation model to identify icebergs, and outputs detection masks and bounding boxes for potential tracking.

🚀 Features

Synthetic Data Generation
Creates RGB satellite-like ocean backgrounds with random iceberg shapes, rotation, shading, and noise.

Deep Learning Pipeline (TensorFlow/Keras)
Implements a lightweight U-Net model for iceberg segmentation.

Augmentation & Training
Random flips and rotations for data diversity.

Evaluation & Visualization
Computes mean Intersection-over-Union (IoU) and saves predicted masks as images.

Tracking Support
Extracts bounding boxes from predicted masks for iceberg location tracking.

📦 Requirements

Install dependencies:

pip install numpy pillow matplotlib scikit-learn tensorflow scipy

🧠 How to Run

Download the Python script:
iceberg_pipeline_tf.py

Run locally or in Google Colab:

python iceberg_pipeline_tf.py


The script will:

Generate a synthetic dataset (N_SAMPLES = 1200)

Train the U-Net model

Save outputs under the outputs/ folder:

outputs/samples/ → visualization of predictions

outputs/predicted_boxes.json → bounding box data

iceberg_unet.h5 → trained model file

📊 Outputs
Output Type	Description
.h5	Trained U-Net model
.png	Visualization of predicted iceberg masks
.json	Detected bounding boxes summary
⚙️ Configuration

Modify these variables at the top of the script to customize:

IMG_SIZE = 128
N_SAMPLES = 1200
EPOCHS = 12
BATCH_SIZE = 16
OUTPUT_DIR = "outputs"

🔬 Future Enhancements

Integrate real satellite data (Sentinel-1/2 or MODIS)

Add multi-spectral channels for realism

Use cloud simulation and sun glint modeling

Incorporate temporal tracking of icebergs across frames

🧾 Author

William Tarinabo Email:williamtarinabo@gmail.com.
