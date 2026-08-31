import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog

IMG_SIZE = (128, 128)

FEATURE_NAMES = [
    "corners",
    "aspect_ratio",
    "extent",
    "solidity",
    "circularity"
]


def extract_shapes(image):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Binarize and denoise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Get contours
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No shape was detected in the image")

    # choose largest contour (ignore tiny noise)
    img_area = gray.shape[0] * gray.shape[1]
    min_area = max(10, 0.005 * img_area)
    large_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if large_contours:
        contour = max(large_contours, key=cv2.contourArea)
    else:
        contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    corner_count = len(approximation)

    _, _, width, height = cv2.boundingRect(contour)
    aspect_ratio = float(width) / height if height else 0.0
    area = cv2.contourArea(contour)
    extent = float(area) / (width * height) if width and height else 0.0

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = float(area) / hull_area if hull_area else 0.0

    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter else 0.0

    return [corner_count, aspect_ratio, extent, solidity, circularity]


def extract_hog_emojis(img):
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("RGB"))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    feature_vector = hog(
        img_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
    )
    return feature_vector
