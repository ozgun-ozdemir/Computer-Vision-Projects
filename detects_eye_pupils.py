import cv2
import numpy as np

image = cv2.imread('face.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (fx, fy, fw, fh) in faces:
    face_region = gray[fy:fy+fh, fx:fx+fw]
    face_color = image[fy:fy+fh, fx:fx+fw]

    eyes = eye_cascade.detectMultiScale(face_region)
    for (ex, ey, ew, eh) in eyes:
        eye_region = face_region[ey:ey+eh, ex:ex+ew]
        
        blurred = cv2.medianBlur(eye_region, 5)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                    param1=50, param2=30, minRadius=20, maxRadius=40)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(face_color, (ex + x, ey + y), r, (0, 255, 0), 4)
                cv2.rectangle(face_color, (ex + x - 5, ey + y - 5), (ex + x + 5, ey + y + 5), (0, 128, 255), -1)

cv2.imshow("Eye Pupils Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()