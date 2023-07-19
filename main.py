# CV2 (Open CV) provides a wide range of functions and tools for image and video processing.
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = "Training_images"
images = []
classNames = []
myList = os.listdir(path)  # List of files inside path dir
print(myList)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    # The os.path.splitext() function takes a file path as input and returns a tuple of two strings.
    # The first string is the file name without the extension, and the second string is the file extension.
print(classNames)

# The code converts the color space of the image from BGR (Blue-Green-Red) to RGB (Red-Green-Blue) using cv2.cvtColor().
#  This conversion is necessary because the face_recognition library expects images in RGB format for its face recognition algorithms.

# Face encoding refers to the process of extracting a numerical representation, or encoding, from a face image. This encoding captures the unique features and characteristics of the face, which can be used to identify or compare faces. Face encoding algorithms analyze facial landmarks, patterns, and features to create a compact and discriminative representation of the face.


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open("Attendance.csv", "a+") as f:
        f.seek(0)  # Move the file pointer to the beginning
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(
                entry[0].strip()
            )  # The strip() function is then applied to remove any leading or trailing whitespace characters from the name
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.write(f"\n{name},{dtString}")


encodeListKnown = findEncodings(images)
print("Encoding Complete")
# print(encodeListKnown)

cap = cv2.VideoCapture(0)
# The line cap = cv2.VideoCapture(0) creates a video capture object to capture video from a webcam
# The argument 0 passed to cv2.VideoCapture() specifies the index of the webcam device to be used. In this case, 0 indicates the default webcam device.

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                img,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            markAttendance(name)

    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
