# Shape Recognition

A Python-based machine learning project for recognizing geometric shapes and emoji-like visual classes using feature extraction and classification models.

## Overview

This repository contains a Streamlit web app and supporting scripts for:

- Shape classification (circle, square, rectangle, triangle, pentagon, hexagon, star, ellipse)
- Emoji classification using HOG-based features
- Training and evaluation of multiple classifiers
- Interactive image upload and prediction through a browser UI

## Features

- Synthetic shape generation for training data
- Feature extraction for contour-based shape descriptors
- HOG feature extraction for emoji classification
- Trained models for:
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
  - Support Vector Machine
- Streamlit interface for visualizing predictions and processing steps

## Project Structure

- `shape_recognition.py` — generates synthetic shapes, trains shape models, and saves model files
- `features.py` — feature extraction utilities for shape and emoji data
- `intrfc.py` — Streamlit application for prediction and model visualization
- `emoji_classifier.py` — emoji model training and evaluation logic
- `requirements.txt` — Python dependencies
- `my_emojis.zip` — sample emoji dataset archive

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Shape Recognition

Shape classification identifies and categorizes geometric shapes using contour-based feature extraction and machine learning classifiers.

### Supported Shapes

- Circle
- Square
- Rectangle
- Triangle
- Pentagon
- Hexagon
- Star
- Ellipse

### How It Works

1. Synthetic training data is generated for each shape class.
2. Contour-based geometric descriptors are extracted from the images.
3. A model such as Decision Tree, KNN, Random Forest, or SVM is trained on the features.
4. New uploaded images are classified based on their extracted shape descriptors.
5. The app displays the predicted class and the processing pipeline.

### Training

Run the shape training script:

```bash
python shape_recognition.py
```

This generates synthetic shape data and saves trained model files using `joblib`.

## Emoji Recognition

Emoji recognition uses HOG-based visual features to classify emoji-like images and other stylized symbols.

### How It Works

1. Emoji images are processed to extract HOG features.
2. Those features capture the gradient structure and local shape patterns.
3. A classifier is trained to learn the mapping from feature vectors to emoji classes.
4. Uploaded emoji images are classified by the trained model.
5. The app displays the prediction, confidence, and processing steps.

### Dataset

The project includes a sample emoji dataset in `my_emojis.zip` for experimentation and model training.

### Training

Run the emoji model training script:

```bash
python emoji_classifier.py
```

This trains and evaluates the classifier on the emoji dataset.

## Running the App

Start the Streamlit application:

```bash
streamlit run intrfc.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## Notes

- This project uses synthetic training data for shape recognition.
- The model files are generated during execution with `joblib`.
- The app is designed for local experimentation and demonstration.

## License

This project does not include a specific license file. If you plan to reuse or distribute it, confirm the intended licensing terms before publishing.
