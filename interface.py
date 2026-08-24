import joblib
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import requests
import os
from PIL import Image
from skimage.feature import hog
from skimage.exposure import rescale_intensity
from features import extract_hog_emojis, extract_shapes

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: pink;
    }
    .result {
        padding-top: 20px;
        background-color: lightblue;
        font-size: 32px;
        font-weight: bold;
        color: green;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATHS = {
    "Decision Tree": "decision_tree_final.pkl",
    "K Nearest Neighbors": "knn_final.pkl",
    "Random Forest": "random_forest_final.pkl",
    "Support Vector Machine": "svc_final.pkl",
}

st.sidebar.selectbox(
    "Choose a Project:",
    ["Shape Prediction", "Emoji Prediction"],
    key="project",
)

if st.session_state.project == "Shape Prediction":
    st.sidebar.selectbox(
        "Select the model",
        ["Decision Tree", "K Nearest Neighbors"],
        key="shape_model",
    )
else:
    st.sidebar.selectbox(
        "Select the model",
        ["Random Forest", "Support Vector Machine"],
        key="emoji_model",
    )


@st.cache_resource
def load_model(path):
    return joblib.load(path)

def model_probabilities(model, features):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([features])[0]
        return dict(zip(model.classes_, probabilities))
    return None

def predict_image(image, project, model_name):
    model = load_model(MODEL_PATHS[model_name])
    features = (
        extract_shapes(image)
        if project == "Shape Prediction"
        else extract_hog_emojis(image)
    )
    prediction = model.predict([features])[0]
    return prediction, model_probabilities(model, features)


image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
prediction_tab, processing_tab = st.tabs(["Prediction", "Processing"])

if "in_image" not in st.session_state:
    st.session_state["in_image"] = None

project = st.session_state.project
model_name = (
        st.session_state.shape_model
        if project == "Shape Prediction"
        else st.session_state.emoji_model
    )



with prediction_tab:
    st.title(project)
    st.header(f"Model: {model_name}")
    col1, col2 = st.columns([2,1])
    with col1:
        if image_input is None:
            col2.info("Upload an image to get a prediction.")
        else:
            st.session_state["in_image"] = image_input
            image = Image.open(image_input).convert("RGB")
            col1.image(image, caption="Input image", use_container_width=True)
            try:
                prediction, probabilities = predict_image(image, project, model_name)
                with col2:
                    top_probability = max(probabilities.values()) if probabilities else None
                    st.write(f'{model_name} prediction')
                    st.success(prediction)
                    if top_probability is not None:
                        st.write(f"Probability: {top_probability:.2%}")
            except Exception as e:
                with col2:
                    st.error(f"Prediction failed: {e}")

with processing_tab:
    st.title(project)
    st.header(f"Model: {model_name}")

    uploaded_image = st.session_state["in_image"]
    st.subheader("Processing steps")
    st.markdown(
        """
        1. Upload and display the image in RGB.
        2. Convert the image to grayscale.
        3. Select the feature extraction method based on the task:
           - Shape Prediction: extract contour-based shape features.
           - Emoji Prediction: extract HOG features.
        4. Load the selected trained model.
        5. Predict the label using the extracted features.
        6. Display the intermediate processing images and the final prediction.
        """
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input Image")
        st.write("Uploaded image is a 3-channel RGB image and the starting point for preprocessing.")
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            image_array = np.array(image)
            st.image(image, caption="Input image", width='stretch')
        else:
            st.info("Upload an image to see the input.")

    with c2:
        st.subheader("Gray Image")
        st.write("The image is converted to grayscale because shape and HOG extraction both work best on intensity data.")
        if uploaded_image is not None:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            st.image(gray_image, caption="Processed image", use_container_width=True)

    if uploaded_image is not None:
        if project == "Shape Prediction":
            try:
                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
                _, binary_image = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)

                c1.write("Binary image for identifying contours")
                c1.write("Binarization simplifies the shape so that contours can be detected reliably.")
                c1.image(binary_image, caption="Binary image", width='stretch')

                contours, _ = cv2.findContours(
                    binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                contour_image = image_array.copy()
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)
                c2.write("External contours")
                c2.image(contour_image, caption="External contours", width='stretch')
            except Exception as e:
                c1.error(f"Processing failed: {e}")
                c2.info("Unable to show contours for this image.")

            if model_name == "K Nearest Neighbors":
                c1.header("K Nearest Neighbors model evaluation")
                df = pd.DataFrame({
                    'N_neighbors': [3, 5, 7, 9, 11, 13],
                    'Accuracy': ['0.91', '0.90', '0.89', '0.88', '0.86', '0.85'],
                })
                c1.dataframe(df, width='stretch')
            else:
                c1.header("Decision Tree model evaluation")
                c1.write("The plot below shows validation accuracy for different tree depths.")
                nc1, nc2 = st.columns(2)
                df1 = pd.DataFrame({
                    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    'Accuracy_gini': [0.581, 0.81, 0.898, 0.917, 0.917, 0.924, 0.927, 0.924, 0.919, 0.917, 0.924, 0.922],
                })
                df2 = pd.DataFrame({
                    'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                    'Accuracy_entropy': [0.896, 0.914, 0.919, 0.922, 0.919, 0.922, 0.919, 0.911, 0.904, 0.909, 0.906, 0.901],
                })
                nc1.write('criterion = gini')
                nc1.dataframe(df1, width='stretch')
                nc2.write('criterion = entropy')
                nc2.dataframe(df2, width='stretch')

        else:
            img_resized = cv2.resize(gray_image, (128, 128), interpolation=cv2.INTER_AREA)
            feature_vector, hog_vis = hog(
                img_resized,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True,
            )
            hog_vis = rescale_intensity(hog_vis, in_range=(0, 10), out_range=(0, 255))
            c1.image(hog_vis.astype(np.uint8), caption="HOG Visualization", width='stretch')
            c2.subheader("HOG visualization")
            c2.write("HOG (Histogram of Oriented Gradients) shows which gradient patterns the model uses for emoji classification.")

            if model_name == "Random Forest":
                c1.header("Random Forest model evaluation")
                df = pd.DataFrame({
                    'n_estimators': [1,2,3,4,5,6,7,8,9,10,15,20,30],
                    'Accuracy': [0.47, 0.57, 0.62, 0.69, 0.76, 0.82, 0.85, 0.90, 0.93, 0.94, 0.96, 0.96, 0.95],
                })
                c1.dataframe(df, width='stretch')
            else:
                c1.header("Support Vector Machine model evaluation")
                c1.write("Support Vector Classifier  uses  linear kernel for distinguishing the classes.")
                c2.header(" ")
                c2.header(" ")
                c2.write("Computationally efficient: Linear SVM is generally faster to train and predict than nonlinear kernels such as RBF, especially when the feature vectors are large.")
                c1.write("Lower risk of overfitting: With a relatively small dataset, a linear decision boundary can provide better generalization than an overly flexible nonlinear boundary.")
