# importing Libraries
import cv2
import dlib
import numpy as np

#Start Video Capture

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks1.dat")

while True:
    _, Frame = cap.read()
    Frame = cv2.flip(Frame, 1)
    gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    FaceBoundingBoxes = detector(gray)
    for FaceBoundingBox in FaceBoundingBoxes:
        x1 = FaceBoundingBox.left()
        x2 = FaceBoundingBox.right()
        y1 = FaceBoundingBox.top()
        y2 = FaceBoundingBox.bottom()
        #cv2.rectangle(Frame, (x1,y1), (x2,y2), (0,255,0), 3)
        landmarks = predictor(gray, FaceBoundingBox)
        for n in range(0,68):            
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(Frame, (x,y), 3, (0,255,0), -1)
        
    cv2.imshow("Frame", Frame)
    
    key = cv2.waitKey(1)
    if key==27:
        break
# close all windows
cv2.destroyAllWindows()