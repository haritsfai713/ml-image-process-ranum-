import cv2
import numpy as np

# Load the image
image = cv2.imread('mango-test3.jpg')

# Convert image from BGR to HSV for color-based segmentation
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define thresholds to exclude white and shadow colors
# White: high value and low saturation
white_lower = np.array([0, 0, 6])   # Lower bound for white in HSV
white_upper = np.array([180, 50, 255]) # Upper bound for white

# Shadows: low brightness
shadow_lower = np.array([0, 0, 0])     # Lower bound for shadows (low value)
shadow_upper = np.array([180, 255, 50]) # Upper bound for shadows

# Create masks for white and shadow
white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
shadow_mask = cv2.inRange(hsv_image, shadow_lower, shadow_upper)

# Invert the masks to select areas that are NOT white or shadows
non_white_mask = cv2.bitwise_not(white_mask)
non_shadow_mask = cv2.bitwise_not(shadow_mask)

# Combine the two masks to exclude both white and shadow areas
combined_mask = cv2.bitwise_and(non_white_mask, non_shadow_mask)

# Apply the mask to the mango region
mango_region = cv2.bitwise_and(hsv_image, hsv_image, mask=combined_mask)

# Reshape the mango region into a 2D array of HSV values
hsv_values = mango_region.reshape((-1, 3))

# Remove zero HSV values (background pixels)
hsv_values = hsv_values[np.all(hsv_values != [0, 0, 0], axis=1)]

# Find the lowest and highest HSV values
lowest_hsv = np.min(hsv_values, axis=0)
highest_hsv = np.max(hsv_values, axis=0)

# Print the results
print(f'Lowest HSV: {lowest_hsv}')
print(f'Highest HSV: {highest_hsv}')
