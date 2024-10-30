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
    # plt.imshow(foreground_rgb)
    # plt.axis('off')
    # plt.show()
    
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
    # plt.imshow(foreground_rgb)
    # plt.axis('off')
    # plt.show()
    cv2.imshow('CV2 display',foreground_rgb)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    
    return foreground

def extract_mango_colors(image_rmvbg, n_clusters=3):
    # Load the image
    image = image_rmvbg
    
    # Convert to RGB (since OpenCV loads in BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Create a mask to exclude black pixels
    mask = np.all(pixels != [0, 0, 0], axis=1)  # Black color condition
    filtered_pixels = pixels[mask]  # Apply the mask
    
    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(filtered_pixels)

    # Get the dominant color
    dominant_color = kmeans.cluster_centers_[0]
    # Convert to RGB integer format
    dominant_color = dominant_color.astype(int)
    dominant_color_rgb = tuple(dominant_color)
    
    # Get the color clusters (centroids) and labels for each pixel
    # cluster_centers = kmeans.cluster_centers_.astype(int)  # RGB values of cluster centers
    # labels = kmeans.labels_
    
    # return cluster_centers, labels, filtered_pixels
    return dominant_color_rgb

def calculate_cluster_proportions(labels, n_clusters):
    # Count the pixels in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    
    return proportions


def resize_image(image_file):
    # Load the image
    image = image_file
    
    # Get your screen resolution (you can adjust this based on your screen size)
    screen_width = 1920  # Example: width of the screen
    screen_height = 1080  # Example: height of the screen
    
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Calculate the scale factor while maintaining the aspect ratio
    scale_width = screen_width / width
    scale_height = screen_height / height
    scale = min(scale_width, scale_height)
    
    # Resize the image to fit within the screen
    new_width = int(width * scale*0.7)
    new_height = int(height * scale*0.7)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized_image
    # Display the resized image
    # cv2.imshow('Resized Image', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def calculate_aspect_ratio(image_file):
    # Read the image
    # image = cv2.imread(image_file)
    # image = image_file

    resized_image = resize_image(image_file)
    # resized_image = image_file

    # Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    
    # Apply a threshold to get a binary image
    # _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                cv2.THRESH_BINARY_INV, 11, 2)
    # Apply a threshold to separate the mango from the black background
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Threshold > 0 to ignore black background
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours to locate the mango
    contours1, _1 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    cv2.imshow('Canny Edges After Contouring', edged) 
    cv2.waitKey(0) 
    
    # print("Number of Contours found = " + str(len(contours1))) 
    
    # Draw all contours 
    # -1 signifies drawing all contours 
    cv2.drawContours(resized_image, contours1, -1, (0, 255, 50), 3) 
    
    cv2.imshow('Contours', resized_image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    
    # if len(contours1) == 0:
    #     print("No contour found!")
    #     return None
    
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
    output_image = resized_image.copy()
    cv2.ellipse(output_image, ellipse, (0, 255, 0), 2)

    # Calculate the endpoints of the major axis
    major_axis_length = major_axis
    minor_axis_length = minor_axis

    # Convert angle to radians
    theta = np.radians(angle)

    # Endpoints of the major axis
    major_axis_x1 = int(center[0] + major_axis_length * np.cos(theta))
    major_axis_y1 = int(center[1] + major_axis_length * np.sin(theta))
    major_axis_x2 = int(center[0] - major_axis_length * np.cos(theta))
    major_axis_y2 = int(center[1] - major_axis_length * np.sin(theta))

    # Endpoints of the minor axis
    minor_axis_x1 = int(center[0] + minor_axis_length * np.cos(theta + np.pi / 2))
    minor_axis_y1 = int(center[1] + minor_axis_length * np.sin(theta + np.pi / 2))
    minor_axis_x2 = int(center[0] - minor_axis_length * np.cos(theta + np.pi / 2))
    minor_axis_y2 = int(center[1] - minor_axis_length * np.sin(theta + np.pi / 2))

    # Draw the major axis (blue line)
    cv2.line(output_image, (major_axis_x1, major_axis_y1), (major_axis_x2, major_axis_y2), (255, 0, 0), 2)

    # Draw the minor axis (green line)
    cv2.line(output_image, (minor_axis_x1, minor_axis_y1), (minor_axis_x2, minor_axis_y2), (100, 255, 0), 2)
    
    # Show the result (optional)
    cv2.imshow('Fitted Ellipse', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # print(f"Aspect Ratio: {aspect_ratio}")
    return aspect_ratio


image_filename = f"D:/ITB/OneDrive - Institut Teknologi Bandung/Akademik/RA/Captured Dataset/Whole Pic/15-10-2024/mango-2-a.jpg"       
x, y, w, h, img_with_bbox = load_yolo(image_filename)
if x is not None:
    print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
    foreground = apply_grabcut(image_filename, (x, y, w, h))
    # You can now use this bounding box for GrabCut
    aspectRatio = calculate_aspect_ratio(foreground)
    # Append the result as a tuple (specimen_num, label, L, a, b)
    # results.append((specimen_num, label, aspectRatio))

else:
    print("Mango detection failed.")


    



