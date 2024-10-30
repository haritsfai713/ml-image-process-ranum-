import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from inference import get_model
import supervision as sv
import os
import pandas as pd

def load_yolo(image_file):
    # define the image url to use for inference
    #image_file = 'mango-test4.jpg'
    image = cv2.imread(image_file)

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


    # # load the results into the supervision Detections api
    # detections = sv.Detections.from_inference(results)

    # # create supervision annotators
    # bounding_box_annotator = sv.BoundingBoxAnnotator()
    # label_annotator = sv.LabelAnnotator()

    # # annotate the image with our inference results
    # annotated_image = bounding_box_annotator.annotate(
    # scene=image, detections=detections)
    # annotated_image = label_annotator.annotate(
    # scene=annotated_image, detections=detections)

    # # display the image
    # sv.plot_image(annotated_image)
    

# Function to remove shadows from the image
def remove_shadow(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv2.split(hsv)

    # Apply a threshold to the value channel to detect shadows
    shadow_mask = cv2.inRange(v, 0, 100)  # Tune the threshold based on your image

    # Use morphology to clean the shadow mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Invert the shadow mask to make shadow areas white
    shadow_mask_inv = cv2.bitwise_not(shadow_mask)

    # Apply the mask to the original image
    shadow_removed = cv2.bitwise_and(image, image, mask=shadow_mask_inv)

    return shadow_removed

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



# Use bounding box for GrabCut to isolate mango
def apply_grabcut(image_path, bbox):
    image = cv2.imread(image_path)
    
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
    plt.imshow(foreground_rgb)
    plt.axis('off')
    plt.show()
    
    return white_rmv

def hsv_remove(image):

    # Convert image from BGR to HSV for color-based segmentation
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for mango (adjust this range based on your mango color)
    lower_mango_color = np.array([1, 51, 51])  # Lower HSV bound for mango
    upper_mango_color = np.array([38, 255, 255])  # Upper HSV bound for mango

    # Create a mask to extract mango-like colors
    mask = cv2.inRange(image, lower_mango_color, upper_mango_color)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the masked image back to RGB (for color extraction)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    # Find contours to locate the mango
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    # Create a bounding box around the mango
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the mango region
    mango_region = masked_image_rgb[y:y+h, x:x+w]

    return mango_region



def remove_background(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Create an initial mask for grabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Create background and foreground models (used internally by grabCut)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define a rectangle around the object (manually or automatically)
    # This rectangle should cover the mango; adjust coordinates based on your image
    height, width = image.shape[:2]
    rect = (width/2, height/2, width - 20, height - 20)  # Example coordinates (x, y, width, height)
    
    # Apply the GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask: make all sure foreground and probable foreground pixels white
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Multiply the mask with the image to isolate the foreground (mango)
    foreground = image * mask2[:, :, np.newaxis]
    
    # Convert BGR to RGB for visualization
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    
    # Display the image with the background removed
    plt.imshow(foreground_rgb)
    plt.axis('off')
    plt.show()
    
    return foreground_rgb

def extract_mango_colors(image_path, n_clusters=3):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to RGB (since OpenCV loads in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # Resize the image to speed up processing
    # resized_image = cv2.resize(image_rgb, (100, 100))
    
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)

    # # Create a mask to exclude black pixels
    # mask = np.all(pixels != [0, 0, 0], axis=1)  # Black color condition
    # filtered_pixels = pixels[mask]  # Apply the mask
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]
    # Convert to RGB integer format
    dominant_color = dominant_color.astype(int)
    dominant_color_rgb = tuple(dominant_color)
    
    # Convert the RGB tuple to a NumPy array and reshape it to 1x1x3 image (1 pixel)
    rgb_pixel = np.array([[dominant_color_rgb]], dtype=np.uint8)

    # Convert the RGB pixel to LAB using OpenCV
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)

    # Extract the LAB values from the result
    lab_tuple = tuple(lab_pixel[0][0])
    L, A, B = lab_tuple

    # Adjust the A and B channels to make the neutral value 0 (by subtracting 128)
    L = (L / 255.0) * 100
    A = A - 128
    B = B - 128

    lab_tuple_adj = (L, A, B)
    
    return lab_tuple_adj

def extract_mango_colors_lab(image_path, n_clusters=3):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to RGB (since OpenCV loads in BGR)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # # Resize the image to speed up processing
    # resized_image = cv2.resize(image_lab, (100, 100))
    
    # Reshape the image to a 2D array of pixels
    pixels = image_lab.reshape(-1, 3)

    # Create a mask to exclude black pixels
    # mask = np.all(pixels != [0, 0, 0], axis=1)  # Black color condition
    # filtered_pixels = pixels[mask]  # Apply the mask
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(pixels)

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
    

def calculate_cluster_proportions(labels, n_clusters):
    # Count the pixels in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    
    return proportions


def calculate_aspect_ratio(image_file):
    # Read the image
    image = cv2.imread(image_file)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a threshold to get a binary image
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contour found!")
        return None
    
    # Assume the largest contour is the mango
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit an ellipse to the largest contour
    if len(largest_contour) < 5:
        print("Not enough points to fit an ellipse.")
        return None
    
    ellipse = cv2.fitEllipse(largest_contour)
    
    # The ellipse will give us the center (x, y), axes lengths (major and minor), and angle
    (center, (major_axis, minor_axis), angle) = ellipse
    
    # Calculate the aspect ratio (major/minor axis length)
    aspect_ratio = major_axis / minor_axis
    
    # Draw the ellipse on the original image for visualization (optional)
    output_image = image.copy()
    cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)
    
    # Show the result (optional)
    cv2.imshow('Fitted Ellipse', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print(f"Aspect Ratio: {aspect_ratio}")
    return aspect_ratio

# Usage
# image_path = "C:/Users/user/Documents/python-code/Fruit-classification/captured-dataset/cropped/crp-mango-1-b.jpg"  # Replace with your mango image path
# cluster_centers = extract_mango_colors(image_path, n_clusters=5)
# print("Cluster Centers (RGB):", cluster_centers)

# List to store the RGB results
results = []

# Loop through the image set (for example, 10 mango specimens, each with two images: 'a' and 'b')
for specimen_num in range(1, 26):  # Change 11 to the number of mango specimens you have
    for label in ['a', 'b']:
        # Generate the filename according to the pattern "mango-[specimen_num]-[label].jpg"
        # image_filename = f"C:/Users/user/Documents/python-code/Fruit-classification/captured-dataset/cropped/crp-mango-{specimen_num}-{label}.jpg"
        image_filename = f"D:/ITB/OneDrive - Institut Teknologi Bandung/Akademik/RA/Captured Dataset/Cropped Section/17-10-2024/crp-mango-{specimen_num}-{label}.jpg"
        
        # Check if the file exists in the current directory or a specified path
        if os.path.exists(image_filename):
            # Call the function to extract the RGB values
            lab_values = extract_mango_colors_lab(image_filename,n_clusters=5)
            
            # Append the result as a tuple (specimen_num, label, L, a, b)
            results.append((specimen_num, label, *lab_values))
        else:
            print(f"File {image_filename} not found!")

# Convert the results to a DataFrame for easy exporting to Excel
df = pd.DataFrame(results, columns=['Specimen Number', 'Label', 'L', 'a', 'b'])

# Save the DataFrame to an Excel file
output_filename = "mango_color_results.xlsx"
df.to_excel(output_filename, index=False)

print(f"LAB values saved to {output_filename}")


    