import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        #reigon of interest = roi
        #[y:y+h,x:x+w] find the box of the face and the imwrite writes the image in the png
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        #How to recognize? deep learned model predict keras, tensorflow, pytorch ,scikit, learn

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        #creating rectangle for around face lol
        color = (255,0,0 )#Blue Green Red not RGB
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y), color, stroke)


    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
