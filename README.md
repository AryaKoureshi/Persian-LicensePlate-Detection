# Persian-LicensePlate-Detection

A project demonstrating how to detect, split, and classify Persian license plate characters using YOLO (for plate/character detection) and a Keras/TensorFlow model (for classification). It includes preprocessing steps, model training, evaluation, and visualization.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Notebook Sections](#notebook-sections)
    - [1. Installing Dependencies](#1-installing-dependencies)
    - [2. Splitting Train & Validation Data](#2-splitting-train--validation-data)
    - [3. Preprocessing & Resizing Data](#3-preprocessing--resizing-data)
    - [4. Visualizing Random Samples](#4-visualizing-random-samples)
    - [5. Building & Training the CNN Model](#5-building--training-the-cnn-model)
    - [6. Plotting Training Accuracy & Loss](#6-plotting-training-accuracy--loss)
    - [7. Evaluation & Confusion Matrix](#7-evaluation--confusion-matrix)
    - [8. YOLO Plate Detection](#8-yolo-plate-detection)
    - [9. YOLO Character Detection](#9-yolo-character-detection)
    - [10. Final Character Classification](#10-final-character-classification)
5. [License](#license)

---

## Introduction

This repository demonstrates how to:
- Split a dataset of images (train/val) using **split-folders**.
- Apply multiple image preprocessing steps (e.g., CLAHE, Otsu threshold, morphological operations) to ensure high-quality character extraction.
- Build and train a CNN for Persian characters and digits (28 classes total).
- Use pre-trained YOLOv8 models for:
  - Detecting license plates on a given image.
  - Detecting individual characters within the license plate.
- Combine everything into a pipeline that outputs the recognized license plate string.

---


## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryaKoureshi/Persian-LicensePlate-Detection.git
   cd Persian-LicensePlate-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   - Ideally, use a **virtual environment** (e.g., `venv` or `conda`) to keep your system clean.

3. **Open the Notebook**:
   - You can run this project using Jupyter Notebook, JupyterLab, or Google Colab.
   - In your terminal, type:
     ```bash
     jupyter notebook PersianLicensePlate.ipynb
     ```
     (Adjust the path as necessary.)

---

## Notebook Sections

### 1. Installing Dependencies

The Notebook starts by installing packages like `split-folders` (for data splitting) and `ultralytics` (for YOLO).

### 2. Splitting Train & Validation Data

Utilizes `splitfolders.ratio` to divide your dataset (e.g., Persian character images) into `train` and `val` directories at an 80/20 ratio.

### 3. Preprocessing & Resizing Data

- Explains the `custom_preprocessing` function, which:
  - Converts to grayscale
  - Applies CLAHE
  - Otsu thresholding
  - Morphological opening
  - Ensures black foreground on white background
  - Resizes images (64x64)
  - Scales to [0,1]

### 4. Visualizing Random Samples

- Randomly loads some images from the training set
- Shows them before and after preprocessing

### 5. Building & Training the CNN Model

- Uses TensorFlow/Keras to define a sequential CNN:
  - Two convolutional + pooling blocks
  - Dense layers
  - Softmax output for the 28 possible classes
- Compiles the model with `Adam` optimizer, trains for a specified number of epochs
- Displays training and validation accuracy/loss

### 6. Plotting Training Accuracy & Loss

- Shows the training history with side-by-side plots for `accuracy` and `loss`
- Evaluates final performance on the validation set

### 7. Evaluation & Confusion Matrix

- Prints the classification report (precision, recall, F1-score)
- Shows the confusion matrix for all 28 classes (digits + letters)

### 8. YOLO Plate Detection

- Uses a specialized YOLO model (`best.pt`) for detecting the license plate region in a test image
- Draws bounding boxes around the detected plate
- Crops the detected plate

### 9. YOLO Character Detection

- Loads another YOLO model (`best_chars.pt`) for detecting each character within the cropped plate
- Sorts the detected characters based on their x-coordinate
- Crops and displays each character

### 10. Final Character Classification

- Applies the same custom preprocessing pipeline to each cropped character
- Uses the trained CNN model to predict each character
- Prints the predicted character with confidence

---

## Results

- **High Accuracy**: The model typically achieves over **99% accuracy** on the validation set.
- **Robust Preprocessing**: Ensures consistent black-on-white character images, boosting classifier performance.
- **YOLO-based Pipeline**: Efficiently detects plates and characters for real-world applications.

---

## License

Feel free to use or modify this project under an open-source license of your choice (e.g. MIT License). Replace this section with your actual License text if needed.

---

**Enjoy detecting and recognizing Persian license plates!**  
If you have questions, please open an issue or pull request.
