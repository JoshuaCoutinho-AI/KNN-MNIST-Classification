# Import necessary libraries
import numpy as np  # Library for numerical computing and handling arrays
import pandas as pd  # Library for data manipulation and analysis
import matplotlib.pyplot as plt  # Library for creating visualizations (e.g., graphs)
from sklearn.model_selection import train_test_split  # Function to split datasets into training and testing sets
from sklearn.preprocessing import StandardScaler  # Function to standardize/scale features (mean = 0, variance = 1)
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors (KNN) classification algorithm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Functions for evaluating models
import tensorflow as tf #TensorFlow is an open-source deep learning framework developed by Google.PyTorch is an open-source deep learning framework developed by Meta (Facebook).

# Load the MNIST dataset
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()
# The MNIST dataset is split into training and testing sets.
# X_train_full and X_test_full contain the image data (28x28 pixel images).
# y_train_full and y_test_full contain the corresponding labels (digit values from 0-9).

# Flatten the images (28x28 -> 784)
X_train_full = X_train_full.reshape(X_train_full.shape[0], -1)
X_test_full = X_test_full.reshape(X_test_full.shape[0], -1)
#(X_train_full.shape[0], -1) keeps the number of samples the same (e.g., 60000 images).1 automatically calculates the remaining dimension (e.g., 784 features per image)
# Each image in MNIST is 28x28 pixels, so we flatten it into a 1D vector of length 784 (28 * 28).
# This step converts the images into a format suitable for machine learning algorithms, which expect 1D arrays.

# Normalize the data (scale pixel values to range 0-1)
X_train_full = X_train_full / 255.0
X_test_full = X_test_full / 255.0
# The pixel values in MNIST range from 0 to 255. By dividing by 255.0, we scale the values to the range [0, 1].
# This helps the model converge faster and improves performance.

# Split training data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
# We split the training data (X_train_full and y_train_full) into two subsets:
# - 80% for training (X_train, y_train)
# - 20% for validation (X_valid, y_valid)
# random_state=42 ensures that the split is reproducible (same every time).

# Standardize the data
scaler = StandardScaler()  # Instantiate a StandardScaler object to standardize data.
X_train = scaler.fit_transform(X_train)  # Fit the scaler to the training data and transform it.
X_valid = scaler.transform(X_valid)  # Use the already fitted scaler to standardize the validation data.
X_test = scaler.transform(X_test_full)  # Standardize the test data using the fitted scaler.
# Standardization means the data will have a mean of 0 and a standard deviation of 1, which helps many machine learning algorithms perform better.

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # Instantiate a KNeighborsClassifier with k=5 (5 nearest neighbors).
knn_model.fit(X_train, y_train)  # Train the model on the training data.

# Predict on the test set
y_pred = knn_model.predict(X_test)  # Use the trained KNN model to predict labels for the test set.

# Evaluate the model
print("Classification Report:")  # Print the classification report header.
print(classification_report(y_test_full, y_pred))  # Display precision, recall, f1-score, and support for each class (digit).
# The classification report gives a detailed evaluation of the model's performance on each class.

print("\nConfusion Matrix:")  # Print the confusion matrix header.
print(confusion_matrix(y_test_full, y_pred))  # Display the confusion matrix to see how many instances were correctly/incorrectly classified.
# The confusion matrix shows how many predictions were correct/incorrect for each class.

print("\nAccuracy Score:", accuracy_score(y_test_full, y_pred))  # Calculate the overall accuracy of the model.
# The accuracy score is the proportion of correct predictions to total predictions.

# Visualize some predictions
def plot_predictions(images, labels, predictions, n=10):
    plt.figure(figsize=(15, 4))  # Create a figure with a specific size (15 inches wide, 4 inches tall).
    for i in range(n):  # Loop over the first n images.
        plt.subplot(1, n, i + 1)  # Create a subplot for each image (1 row, n columns).
        plt.imshow(images[i].reshape(28, 28), cmap="gray")  # Reshape the 1D image array back to 28x28 and display it.
        plt.title(f"True: {labels[i]}\nPred: {predictions[i]}")  # Set the title to show true and predicted labels.
        plt.axis("off")  # Hide the axes for a cleaner look.
    plt.show()  # Display the images.

# Plot the first 10 predictions
plot_predictions(X_test_full[:10], y_test_full[:10], y_pred[:10])  # Plot the first 10 images from the test set along with their true and predicted labels.
