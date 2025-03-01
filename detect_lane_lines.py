import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('documents/Computer-Vision-Projects-main/images/road.jpg')

def detect_lane_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 180)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    # Define the region of interest using a polygon
    polygon = np.array([[
        (0, height), 
        (width, height),  
        (width, int(height * 0.5)),
        (0, int(height * 0.5))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Transform to detect lane lines
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=60, maxLineGap=90)

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5) # Draw lines in green

    # Combine the original image with the lines
    result = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    return result

result_image = detect_lane_lines(image)

image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')
plt.show()
