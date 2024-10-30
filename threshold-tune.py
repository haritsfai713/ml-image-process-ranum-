import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to remove shadows from the image
def remove_shadow(image, threshold):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv2.split(hsv)

    # Apply a threshold to the value channel to detect shadows
    shadow_mask = cv2.inRange(v, 0, threshold)

    # Use morphology to clean the shadow mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Invert the shadow mask to make shadow areas white
    shadow_mask_inv = cv2.bitwise_not(shadow_mask)

    # Apply the mask to the original image
    shadow_removed = cv2.bitwise_and(image, image, mask=shadow_mask_inv)

    return shadow_removed, shadow_mask

# Interactive tuning of threshold
def tune_shadow_threshold(image_path):
    image = cv2.imread(image_path)

    # Function to update shadow mask in real time based on slider value
    def update_threshold(val):
        threshold = cv2.getTrackbarPos('Threshold', 'Shadow Removal')
        shadow_removed, shadow_mask = remove_shadow(image, threshold)

        # Display shadow mask and the shadow-removed image side by side
        combined_display = np.hstack((cv2.cvtColor(shadow_mask, cv2.COLOR_GRAY2BGR), shadow_removed))
        cv2.imshow('Shadow Removal', combined_display)

    # Create a window to display the results
    cv2.namedWindow('Shadow Removal')

    # Create a trackbar for the threshold (max value = 255)
    cv2.createTrackbar('Threshold', 'Shadow Removal', 100, 255, update_threshold)

    # Initialize with default threshold
    update_threshold(0)

    # Wait until user presses a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'mango-test4.jpg'  # Replace with your image path
tune_shadow_threshold(image_path)
