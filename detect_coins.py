import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('documents/Computer-Vision-Projects-main/images/coins.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur 
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
# Detect circles using Hough Transform
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
                            param1=100, param2=100, minRadius=70, maxRadius=180)

num_coins = 0
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    num_coins = len(circles)
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 5) # Draw the circle in green
        cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), (0, 128, 255), -1)  #Draw an orange rectangle at the center of the circle

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title(f"Coins Detected: {num_coins}")
plt.axis('off')  
plt.show()