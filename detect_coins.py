import cv2
import numpy as np

image = cv2.imread('coins.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 11), 0)

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=90,
                            param1=50, param2=30, minRadius=40, maxRadius=80)

num_coins = 0
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    num_coins = len(circles)
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 10, y - 10), (x + 10, y + 10), (0, 128, 255), -1)


cv2.putText(image, f"Coins: {num_coins}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Coins Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()