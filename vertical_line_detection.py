
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


# Load the image
image_path = 'C:/Python-STL/letters/letter_sample_2D-1a.png'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('Grayscale Image', gray)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()

# Apply adaptive thresholding to get a binary image
block_size = 31  # Must be an odd number
C = 20  # Constant subtracted from the mean
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)

cv2.imshow('Binary Image', binary)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# Apply morphological operations to enhance the binary image
# Apply morphological operations to enhance the binary image
kernel_size = (2, 2)  # Adjust the kernel size
kernel = np.ones(kernel_size, np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Display the binary image
cv2.imshow('Binary Image', binary)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()

# Apply connected component analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Create a directory to save the extracted characters
output_dir = 'extracted_characters'
os.makedirs(output_dir, exist_ok=True)

# Initialize a counter for character images
char_count = 0

# Filter and save each character
for i in range(1, num_labels):  # Start from 1 to skip the background
    x, y, w, h, area = stats[i]
    # Filter out small components
    if w > 10 and h > 10 and area > 50:  # Adjust the size and area thresholds as needed
        char_roi = binary[y:y+h, x:x+w]

        # Save the character image
        char_image_path = os.path.join(output_dir, f'char_{char_count}.png')
        cv2.imwrite(char_image_path, char_roi)
        char_count += 1

        # Draw bounding box around the character (optional for visualization)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the result (optional for visualization)
cv2.imshow('Detected Characters', image)
cv2.waitKey(0)
cv2.destroyAllWindows()









# # Load and binarize the image
# img = cv2.imread('C:/Python-STL/letters/letter_sample_2D-1a_gray.png', cv2.IMREAD_GRAYSCALE)
# _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# if img is None:
#     print("Error: Image not found or unable to load.")
#     exit()

# # Apply edge detection (Canny)
# edges = cv2.Canny(img, 50, 150, apertureSize=3)

# # Apply Hough Line Transform
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# # Check if any lines are detected
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         # Draw the lines on the original image
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# else:
#     print("No lines detected.")

# # Display the result
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Detected Vertical Lines')
# plt.show()


# # Sum pixels vertically to find text columns (vertical projection)
# vertical_sum = np.sum(binary_img, axis=0)
# threshold = np.max(vertical_sum) * 0.2  # Tune the threshold as needed

# # Find start and end of each vertical line
# vertical_lines = []
# in_line = False
# for i, value in enumerate(vertical_sum):
#     if value > threshold and not in_line:
#         start = i
#         in_line = True
#     elif value <= threshold and in_line:
#         end = i
#         in_line = False
#         vertical_lines.append((start, end))

# # Create output directory for each vertical line
# output_dir = 'C:/Python-STL/letters/segmented_lines'
# os.makedirs(output_dir, exist_ok=True)

# # Segment each vertical line from the image and save it
# for i, (start, end) in enumerate(vertical_lines):
#     line_img = binary_img[:, start:end]
#     cv2.imwrite(f'{output_dir}/line_{i}.png', line_img)
