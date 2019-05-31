import cv2
import matplotlib.pyplot as plt
#import dlib
#from imutils import face_utiles
font=cv2.FONT_HERSHEY_SIMPLEX
#load image
gray= cv2.imread('messi.jpg',0)
plt.figure(figsize=(12,8))
plt.imshow(gray,cmap='gray')
plt.show()
# Create the haarcascade
##############
#加载opcv训练模型
cascPath = "C:/Program Files/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eyePath = "C:/Program Files/Python36/Lib/site-packages/cv2/data/haarcascade_eye.xml"
smilePath = "C:/Program Files/Python36/Lib/site-packages/cv2/data/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)
##############
"""
#image
#Detect faces
faces = faceCascade.detectMultiScale(
gray,
scaleFactor=1.1,
minNeighbors=5,
flags=cv2.CASCADE_SCALE_IMAGE
)
# For each face
for (x, y, w, h) in faces:
    # Draw rectangle around the face
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 255, 255), 3)

plt.figure(figsize=(12,8))
plt.imshow(gray, cmap='gray')
plt.show()
"""
#####################
# Launch Video Capture
video_capture = cv2.VideoCapture(0)

# While letter "q" not pressed
while True:

    # Capture video frame-by-frame
    ret, frame = video_capture.read()

    # Transform to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        # minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # For each face
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(frame, 'Face', (x, y), font, 2, (255, 0, 0), 5)

        # Crop image to the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect the mouth
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.16,
            minNeighbors=40,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # Put a rectangle and text around mouth
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            cv2.putText(frame, 'Smile', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

        # Detect the eyes
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            minSize=(10, 10),
            minNeighbors=20)

        # Put a rectangle and text around each exe
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv2.putText(frame, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)

        # Count the number of faces
    cv2.putText(frame, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)

    # Display the video output
    cv2.imshow('Video', frame)

    # Quit video by typing Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Capture
video_capture.release()
cv2.destroyAllWindows()