import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('documents/Computer-Vision-Projects-main/images/face.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(70, 70))

for (fx, fy, fw, fh) in faces:
    face_region = gray[fy:fy+fh, fx:fx+fw]
    face_color = image[fy:fy+fh, fx:fx+fw]

    # Detect eyes 
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.2, minNeighbors=7, minSize=(20, 20))

    for (ex, ey, ew, eh) in eyes:
        eye_region = face_region[ey:ey+eh, ex:ex+ew]
        
        # Apply median blur to the eye region
        blurred = cv2.medianBlur(eye_region, (7), 0)

        # Applying Hough Circle Transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=200,
                                    param1=20, param2=20, minRadius=20, maxRadius=60)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(face_color, (ex + x, ey + y), r, (0, 255, 0), 5) # Drawing a green circle around the eye
                cv2.rectangle(face_color, (ex + x - 5, ey + y - 5), (ex + x + 5, ey + y + 5), (0, 128, 255), -1) # Drawing an orange rectangle at the center of the eye

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off')  
plt.show()
