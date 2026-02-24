📌 Overview

This project focuses on improving the reliability of AI-based medical image classification by integrating Clinical Rule-Based Post-Processing with a Cost-Aware Thresholding mechanism on top of a CNN-based skin lesion detection model.

A pretrained EfficientNetB0 Convolutional Neural Network (CNN) is used to classify dermoscopic skin lesion images as:

Benign (0) – Non-cancerous

Malignant (1) – Cancerous

In addition to image-based predictions, clinical metadata such as lesion size is incorporated to refine the model’s output using a rule-based engine. This helps reduce false positives and improves diagnostic reliability.

🎯 Objective

To reduce false positives in AI-based medical diagnosis by combining:

Image-based deep learning predictions

Clinical metadata-driven rule filtering

Cost-aware decision threshold optimization

📂 Dataset

Source: ISIC (International Skin Imaging Collaboration) Dataset

Data Used:

Dermoscopic skin lesion images

Clinical metadata (clin_size_long_diam_mm)

Balanced dataset:

45 Malignant samples

45 Benign samples

🧠 Methodology
1. Data Preprocessing

Image resizing to 224×224 pixels

Pixel normalization

Class balancing

Metadata cleaning & imputation

Image augmentation (rotation, flip)

2. Model Development

Transfer Learning using EfficientNetB0

Base model layers frozen

Custom classification head added:

Global Average Pooling

Dense Layer (ReLU)

Dense Output Layer (Sigmoid)

3. Training Configuration

Optimizer: Adam

Learning Rate: 0.0001

Loss Function: Binary Crossentropy

Batch Size: 8

Epochs: 5

4. Clinical Rule-Based Post-Processing

If:

clin_size_long_diam_mm < 2 mm

Then:

Prediction probability → 0 (Benign)

This suppresses clinically insignificant false positives.

5. Cost-Aware Thresholding

Thresholds from 0 to 1 are evaluated to minimize:

Total Cost = (C_FP × FP) + (C_FN × FN)

Best threshold selected for optimal clinical safety.

📊 Results
Model	Accuracy Before	Accuracy After	FP Reduction
CNN (EfficientNetB0)	54%	85%	6 → 2
Metadata Model	61%	83%	7 → 3

Performance evaluated using:

Accuracy

Confusion Matrix

Heatmaps

🛠️ Tech Stack

Python

TensorFlow / Keras

EfficientNetB0

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

⚠️ Limitations

Binary classification only

No real-time deployment

Static rule engine

Limited dataset size

Lacks explainability (Grad-CAM not implemented)

🚀 Future Scope

Multi-class lesion classification

Integration of patient demographics

Explainable AI techniques (Grad-CAM)

Real-time mobile deployment

Adaptive rule-learning system
