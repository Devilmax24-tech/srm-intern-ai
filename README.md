# 🖼️ CIFAR-10 Image Classification
## Deep Learning Project - SRM Intern

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-86.71%25-brightgreen)]()

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Train/Validation Split](#trainvalidation-split)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Metrics Reported](#metrics-reported)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## 🎯 Project Overview

This project implements two CNN architectures for CIFAR-10 image classification:
- **Baseline CNN**: Simple 3-layer convolutional network
- **Improved CNN**: Deep network with augmentation + regularization

**Best Result:** `86.71%` Test Accuracy (Improved CNN)

---

## 📊 Dataset

| Description | Value |
|------------|-------|
| **Classes** | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| **Image Size** | 32x32x3 (RGB) |
| **Total Images** | 60,000 |
| **Training Set** | 50,000 |
| **Test Set** | 10,000 |

---

## ✂️ Train/Validation Split

python
from sklearn.model_selection import train_test_split

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Split training data: 80% train, 20% validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,        # 20% for validation
    random_state=42,      # Reproducible results
    stratify=y_train      # Equal class distribution
)

<img width="729" height="322" alt="Screenshot From 2026-02-11 22-01-02" src="https://github.com/user-attachments/assets/d130b379-0c5c-41dd-b084-951bb3ba2f45" />

🏛️ Model Architectures

1️⃣ Baseline CNN

<img width="197" height="316" alt="Screenshot From 2026-02-11 22-29-11" src="https://github.com/user-attachments/assets/46211c8d-55b8-4d7e-8a7b-fe6faadd7bc7" />
<img width="197" height="316" alt="Screenshot From 2026-02-11 22-29-11" src="https://github.com/user-attachments/assets/77ef138f-6ec9-42de-81fc-79378a053b17" />
<img width="204" height="142" alt="Screenshot From 2026-02-11 22-29-42" src="https://github.com/user-attachments/assets/55cfc3c8-c0b7-4a8e-8015-399cbae3163a" />



🏆 Results


🥇 Best Model: Improved CNN

<img width="378" height="227" alt="Screenshot From 2026-02-11 22-31-53" src="https://github.com/user-attachments/assets/d0a93955-f322-4b35-928c-729c9ae89596" />


<img width="541" height="408" alt="Screenshot From 2026-02-11 22-14-57" src="https://github.com/user-attachments/assets/0810f54c-9754-474c-867e-80b085555f7e" />


<img width="514" height="482" alt="Screenshot From 2026-02-11 22-16-16" src="https://github.com/user-attachments/assets/88323bf1-6e8b-42b8-bd57-557a08178738" />


📉 Confusion Matrix Summary
          Predicted
        A  Au B  C  D  Do F  H  S  T  ← Actual
        ┌─────────────────────────┐
     A  │88 1  2  0  0  0  0  0  6  1│
     Au │1 93  0  0  0  0  0  0  2  2│
     B  │2  0 81  4  4  3  2  2  1  1│
     C  │1  0  3 79  3  7  2  2  1  1│
     D  │0  0  4  3 83  3  4  2  0  0│
     Do │0  0  4  7  3 80  2  3  0  1│
     F  │1  0  3  3  2  2 90  1  0  0│
     H  │0  0  2  2  3  3  2 87  0  1│
     S  │5  1  1  0  0  0  0  0 91  1│
     T  │2  2  0  0  0  0  0  0  1 92│
        └─────────────────────────┘


📏 Metrics Reported

<img width="676" height="336" alt="Screenshot From 2026-02-11 22-17-41" src="https://github.com/user-attachments/assets/9db35755-1e8b-4959-9e3c-b0d440521e6c" />

<img width="548" height="362" alt="Screenshot From 2026-02-11 22-18-28" src="https://github.com/user-attachments/assets/2f331c06-49bd-4108-a03b-dc029e1adf28" />

💻 Installation

Requirements:
Python 3.8+
TensorFlow 2.8+
4GB+ RAM

Quick Setup
# Clone repository
git clone https://github.com/yourusername/cifar10-classification.git
cd cifar10-classification

# Install dependencies
pip install -r requirements.txt

requirements.txt
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=1.0.0
jupyter>=1.0.0

🚀 Usage

1️⃣ Training Baseline Model
python train_baseline.py

2️⃣ Training Improved Model
python train_improved.py

3️⃣ Evaluate Models
python evaluate.py

4️⃣ Quick Prediction

from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('models/improved_cifar10_model.h5')

# Predict
predictions = model.predict(image)
class_idx = np.argmax(predictions)
confidence = np.max(predictions)

print(f"Predicted: {class_names[class_idx]} ({confidence:.2%})")


🔑 Key Improvements Summary

<img width="470" height="234" alt="Screenshot From 2026-02-11 22-27-28" src="https://github.com/user-attachments/assets/38787961-5e37-417a-9f18-f9a816195074" />


📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
