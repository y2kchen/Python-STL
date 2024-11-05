
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Load and binarize the image
img = cv2.imread('C:/Python-STL/letters/letter_sample_2D-1a_gray.png', cv2.IMREAD_GRAYSCALE)
_, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

if img is None:
    print("Error: Image not found or unable to load.")
    exit()

# Apply edge detection (Canny)
edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Check if any lines are detected
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw the lines on the original image
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
else:
    print("No lines detected.")

# Display the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Detected Vertical Lines')
plt.show()


# Sum pixels vertically to find text columns (vertical projection)
vertical_sum = np.sum(binary_img, axis=0)
threshold = np.max(vertical_sum) * 0.2  # Tune the threshold as needed

# Find start and end of each vertical line
vertical_lines = []
in_line = False
for i, value in enumerate(vertical_sum):
    if value > threshold and not in_line:
        start = i
        in_line = True
    elif value <= threshold and in_line:
        end = i
        in_line = False
        vertical_lines.append((start, end))

# Create output directory for each vertical line
output_dir = 'C:/Python-STL/letters/segmented_lines'
os.makedirs(output_dir, exist_ok=True)

# Segment each vertical line from the image and save it
for i, (start, end) in enumerate(vertical_lines):
    line_img = binary_img[:, start:end]
    cv2.imwrite(f'{output_dir}/line_{i}.png', line_img)
