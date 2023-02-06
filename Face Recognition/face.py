import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    Original_labels = pickle.load(f)
    labels = {v:k for k,v in Original_labels.items()}#invserting the pair so it is proper format

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        #reigon of interest = roi
        #[y:y+h,x:x+w] find the box of the face and the imwrite writes the image in the png
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        #How to recognize? deep learned model predict keras, tensorflow, pytorch ,scikit, learn
        id_, conf = recognizer.predict(roi_gray)

        img_item = "1.png"
        cv2.imwrite(img_item, roi_color)
        if conf >=45 and conf <=100:
            print(conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)#white
            stroke =2
            cv2.putText(frame,name, (x,y),font, 1, color,stroke, cv2.LINE_AA)

        #creating rectangle for around face lol
        color = (255,0,0 )#Blue Green Red not RGB
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # mouth = eye_cascade.detectMultiScale(roi_gray)
        # for (mx,my,mw,mh) in mouth:
        #     cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)


    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
