# Computer Vision Classification - Russian Wildlife Dataset

## 📌 Project Overview
This project focuses on image classification using the **Russian Wildlife Dataset**. The dataset consists of 10 classes of animals and humans, and the objective is to train deep learning models for classification. The project involves training a **CNN from scratch**, **fine-tuning a ResNet-18 model**, and applying **data augmentation** techniques to improve performance.

We used **PyTorch** for model development and **Weights & Biases (wandb)** for logging training metrics, visualizations, and performance analysis.

---
## 🚀 Tasks & Methodology
### 1️⃣ Dataset Preparation & Preprocessing
- Downloaded the **Russian Wildlife Dataset** and map class labels as follows:
  ```
  {'amur leopard': 0, 'amur tiger': 1, 'birds': 2, 'black bear': 3, 'brown bear': 4,
   'dog': 5, 'roe deer': 6, 'sika deer': 7, 'wild boar': 8, 'people': 9}
  ```
- Performed **stratified random split** (80% train, 20% validation)
- Created a **custom PyTorch Dataset class**
- Initialized **Weights & Biases (wandb)** for logging

### 2️⃣ Training a CNN from Scratch
- Implemented a **CNN with 3 convolution layers**:
  - Kernel size: **3×3**, Stride: **1**, Padding: **1**
  - Feature maps: **32, 64, 128**
  - Max pooling layers with different kernel sizes (4×4, 2×2)
- Used **ReLU activation** and a fully connected classification head
- Trained for **10 epochs** using:
  - **Cross-Entropy Loss**
  - **Adam optimizer**
- Logged training & validation **losses, accuracies, and confusion matrix** using wandb
- Evaluated overfitting and misclassifications

### 3️⃣ Fine-Tuning a Pretrained ResNet-18
- Fine-tuned a **ResNet-18 (pre-trained on ImageNet)**
- Trained using the same process as the CNN model
- Extracted feature vectors from the backbone (ResNet-18) and visualized **t-SNE plots** in **2D and 3D**

### 4️⃣ Data Augmentation
- Applied **3 or more data augmentation techniques**
- Visualized 5 augmented images
- Trained the model with augmented data and evaluate performance
- Compared training loss to check if overfitting is reduced

### 5️⃣ Performance Comparison
- Compared the performance of **three models**:
  - CNN from scratch
  - Fine-tuned ResNet-18
  - Augmented model
- Analyzed Accuracy, F1-Score, and overfitting trends

---
## 📊 Results & Analysis
- The performance of each model is evaluated based on:
  - **Accuracy & F1-Score**
  - **Confusion Matrix**
  - **t-SNE feature space visualization**
  - **Impact of data augmentation on overfitting**

---
## 📌 Conclusion
This project explores different approaches to **image classification**, comparing a **custom CNN**, **ResNet-18 fine-tuning**, and **data augmentation** strategies. By logging results with **wandb**, we gain insights into model behavior and improvements. 🚀

