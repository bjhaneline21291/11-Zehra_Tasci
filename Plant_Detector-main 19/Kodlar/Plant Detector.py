import cv2

#  Plant Detector
# adjust the trackbar to get proper result

loc = 'cascade3.xml'  
cam_No = 1                     # 0-primary camera        1-secondary camera
objectName = 'Plant'       
frame_Width= 640                     
frame_Height = 480                  
color= (0,255,0)



cap = cv2.VideoCapture(cam_No)
cap.set(3, frame_Width)
cap.set(4, frame_Height)

def empty(a):
    pass

# TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frame_Width,frame_Height+100)
cv2.createTrackbar("Scale","Result",400,1000,empty)
cv2.createTrackbar("Neig","Result",8,50,empty)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)


cascade = cv2.CascadeClassifier(loc)

while True:
    
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
    cap.set(10, cameraBrightness)
    
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    scaleVal =1 + (cv2.getTrackbarPos("Scale", "Result") /1000)
    neig=cv2.getTrackbarPos("Neig", "Result")
    objects = cascade.detectMultiScale(gray,scaleVal, neig)
    
    for (x,y,w,h) in objects:
        area = w*h
        minArea = cv2.getTrackbarPos("Min Area", "Result")
        if area >minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
            cv2.putText(img,objectName,(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            roi_color = img[y:y+h, x:x+w]

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
