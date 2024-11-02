import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_lane_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width, int(height * 0.6)),
        (0, int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

    line_image = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    result = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    return result

image_path = 'road.jpg'
image = cv2.imread(image_path)

lane_lines_image = detect_lane_lines(image)

lane_lines_image_rgb = cv2.cvtColor(lane_lines_image, cv2.COLOR_BGR2RGB)

plt.imshow(lane_lines_image_rgb)
plt.axis('off')
plt.show()