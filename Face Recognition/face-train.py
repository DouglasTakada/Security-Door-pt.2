import cv2
import os
import pickle
from PIL import Image
import numpy as np


#gets path of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
#if above function says cv2 dont have .face function you gotta do:
#pip install opencv-contrib-python

#training labels
current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #Grabs and prints path to all photos it found in the directory Douglas
            #print(path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1

            id_ = label_ids[label]
            print(label_ids)
            #now using numpy to convert image
            #Pil = Python Image Library
            pil_image = Image.open(path).convert("L") # converts image into gray scale
            #Convert image into number array using numpy. It is taking every pixle value and converting
            #it to numbers in the array
            size = (550,500)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8")
            print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5,minNeighbors =5)

            for(x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
            

#save labels so we can use inside other py script
#going to use pickle for this
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids,f)

#now train item itself makin facial recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
