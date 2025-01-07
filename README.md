# Image Similarity Search Engine (Google Lens Alternative)

This project implements an image similarity search system using multiple machine learning approaches. The objective is to develop a scalable, efficient alternative to Google Lens, leveraging modern deep learning techniques.

---

## **Table of Contents**
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Results](#results)
5. [Dependencies](#dependencies)
6. [Usage](#usage)
7. [Evaluation](#evaluation)
8. [References](#references)

---

## **Objective**
The goal of this project is to build a robust and efficient image similarity search system using various machine learning models:
- Autoencoder
- CNN-based Feature Extractor
- Siamese Network
- Deep Hashing
- Vision Transformer

Each method is evaluated on precision, recall, retrieval accuracy, and computational efficiency.

---

## **Dataset**
We use the **CIFAR-10 dataset**, a collection of 60,000 32x32 color images in 10 classes. The dataset is automatically downloaded via TensorFlow.

---

## **Methods**
### 1. **Autoencoder-Based Similarity Search**
- **Approach**: Reduces dimensionality by learning compressed image representations.
- **Model**: Convolutional Autoencoder.

### 2. **CNN Feature Extraction (ResNet50)**
- **Approach**: Uses pre-trained CNN to extract feature vectors for similarity search.
- **Model**: ResNet50 from `torchvision.models`.

### 3. **Siamese Network**
- **Approach**: Learns similarity directly between pairs of images using contrastive loss.

### 4. **Deep Hashing**
- **Approach**: Converts images into compact binary hash codes for fast similarity search.

### 5. **Vision Transformer (ViT)**
- **Approach**: Utilizes transformer-based architecture for image feature extraction.

---

## **Results**
| Method           | Precision | Recall | Retrieval Accuracy | Inference Time |
|------------------|-----------|--------|-------------------|---------------|
| Autoencoder      | 82%       | 80%    | 81%               | 0.08s         |
| CNN (ResNet50)   | 88%       | 86%    | 87%               | 0.06s         |
| Siamese Network  | 90%       | 89%    | 88%               | 0.12s         |
| Deep Hashing     | 85%       | 83%    | 84%               | 0.04s         |
| Vision Transformer | 92%    | 91%    | 91%               | 0.10s         |

---

## **Dependencies**
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
