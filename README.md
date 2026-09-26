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

## Running the App

Start the Streamlit application:

```bash
streamlit run intrfc.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How It Works

1. Images are uploaded through the Streamlit interface.
2. Relevant features are extracted depending on the selected project:
   - Shape Prediction: contour-based geometric descriptors
   - Emoji Prediction: HOG features
3. The selected trained model predicts the class.
4. The app shows the prediction, confidence, and processing pipeline.

## Notes

- This project uses synthetic training data for shape recognition.
- The model files are generated during execution with `joblib`.
- The app is designed for local experimentation and demonstration.

## License

This project does not include a specific license file. If you plan to reuse or distribute it, confirm the intended licensing terms before publishing.
