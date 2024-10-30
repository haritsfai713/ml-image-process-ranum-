import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from inference import get_model
import supervision as sv
import os
import pandas as pd
import pickle

def load_yolo(image_file):
    # define the image url to use for inference
    #image_file = 'mango-test4.jpg'
    image = cv2.imread(image_file)
    # image = image_file

    # resized_image = cv2.resize(image, (100, 100))

    # load a pre-trained yolov8n model
    model = get_model(model_id="mangoes-detection-2/1",api_key="iohwl6W5ldIo1lX7ymJA")

    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image)[0]

    # Access the predictions (list of detections)
    predictions = results.predictions

    if predictions:
        for pred in predictions:
            # Get bounding box coordinates for the mango object
            x = int(pred.x - pred.width / 2)
            y = int(pred.y - pred.height / 2)
            w = int(pred.width)
            h = int(pred.height)

            # Draw the bounding box on the image (optional, for visualization)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Return bounding box for GrabCut
            return x, y, w, h, image

    else:
        print("No mango detected.")
        return None, img  # If no mango is detected

# Use bounding box for GrabCut to isolate mango
def apply_grabcut(image_path, bbox):
    image = cv2.imread(image_path)
    # image = image_path
    
    # Remove shadows from the image
    # shadow_free_image = remove_shadow(image)
    shadow_free_image = image

    # Initialize mask for GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Use the bounding box coordinates from YOLO detection
    x, y, w, h = bbox

    # Apply GrabCut using the detected bounding box
    rect = (x, y, x + w, y + h)
    cv2.grabCut(shadow_free_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create the final mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground = shadow_free_image * mask2[:, :, np.newaxis]
    
    # Convert to RGB for display
    # foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    white_rmv = remove_white_background(foreground)
    # hsv_rmv = hsv_remove(white_rmv)
    foreground_rgb = cv2.cvtColor(white_rmv, cv2.COLOR_BGR2RGB)
    
    # Display the final segmented image
    # plt.imshow(foreground_rgb)
    # plt.axis('off')
    # plt.show()
    
    return white_rmv

def remove_white_background(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    # White has low saturation and high value
    lower_white = np.array([0, 0, 100])  # Adjust value based on white intensity
    upper_white = np.array([180, 50, 255])

    # Create a mask for white color
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Use morphology to clean the white mask (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Invert the white mask so that white areas are black and everything else is white
    white_mask_inv = cv2.bitwise_not(white_mask_cleaned)

    # Apply the mask to the original image to remove the white background
    result = cv2.bitwise_and(image, image, mask=white_mask_inv)

    return result

def extract_mango_colors_lab(image_path, n_clusters=3):
    # Load the image
    # image = cv2.imread(image_path)
    image = image_path
    
    # Convert to RGB (since OpenCV loads in BGR)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # # Resize the image to speed up processing
    # resized_image = cv2.resize(image_lab, (100, 100))
    
    # Reshape the image to a 2D array of pixels
    pixels = image_lab.reshape(-1, 3)

    # Create a mask to exclude black pixels
    mask = np.all(pixels != [0, 0, 0], axis=1)  # Black color condition
    filtered_pixels = pixels[mask]  # Apply the mask
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(filtered_pixels)

    # Count the number of pixels assigned to each cluster
    _, counts = np.unique(kmeans.labels_, return_counts=True)

    # Find the index of the largest cluster
    dominant_cluster_index = np.argmax(counts)

    # Get the RGB values of the dominant color
    dominant_color = kmeans.cluster_centers_[dominant_cluster_index].astype(int)
    dominant_color_lab = tuple(dominant_color)
    L, A, B = dominant_color_lab

    # Adjust the A and B channels to make the neutral value 0 (by subtracting 128)
    L = (L / 255.0) * 100
    A = A - 128
    B = B - 128

    lab_tuple_adj = (L, A, B)
    
    return lab_tuple_adj

# Usage
# image_path = "C:/Users/user/Documents/python-code/Fruit-classification/captured-dataset/cropped/crp-mango-1-b.jpg"  # Replace with your mango image path
# cluster_centers = extract_mango_colors(image_path, n_clusters=5)
# print("Cluster Centers (RGB):", cluster_centers)

# Load your trained model
with open("random_forest_mango_model2.pkl", "rb") as model_file:
    model = pickle.load(model_file)

print(type(model))

img = f"D:/ITB/OneDrive - Institut Teknologi Bandung/Akademik/RA/Captured Dataset/Whole Pic/17-10-2024/mango-4-b.jpg"

x, y, w, h, img_with_bbox = load_yolo(img)
if x is not None:
    print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
    foreground = apply_grabcut(img, (x, y, w, h))
else:
    print("Mango detection failed.")

lab_values = extract_mango_colors_lab(foreground,n_clusters=5)

print("Going to prediction")
# Make a prediction using the loaded model
prediction = model.predict([lab_values])[0]
print("prediction ready",prediction)



    