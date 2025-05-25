# KNN-MNIST-Classification

A simple but effective implementation of handwritten digit classification using the **K-Nearest Neighbors (KNN)** algorithm on the **MNIST** dataset. This project leverages `scikit-learn`, `TensorFlow`, and `matplotlib` to train, evaluate, and visualize predictions.

---

## 🧠 What It Does

- Loads and preprocesses the MNIST digit dataset
- Normalizes and scales the data
- Splits into training, validation, and test sets
- Trains a KNN classifier (`k=5`)
- Evaluates model performance with classification report, confusion matrix, and accuracy
- Visualizes predicted vs true labels for a few sample images

---
## 🖼️ Sample Prediction Plot

![Prediction Sample](knn_output.png)

## 📸 Sample Output
Classification Report:
precision recall f1-score support
       0       0.98      0.99      0.98       980
       1       0.97      0.99      0.98      1135
       2       0.97      0.97      0.97      1032
Confusion Matrix:
[[ 976 0 1 0 0 0 2 0 1 0]
[ 0 1124 2 2 0 0 3 1 3 0]
[ 4 2 1003 2 0 0 1 9 11 0]
Accuracy Score: 0.9729


---

## 📦 Installation & Launch Instructions

### Prerequisites
- Python 3.8+
- pip

### Steps to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/JoshuaCoutinho-AI/KNN-MNIST-Classification.git
cd KNN-MNIST-Classification

