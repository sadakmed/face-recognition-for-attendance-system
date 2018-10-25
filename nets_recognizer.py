import cv2
import numpy as np
from A_embedder import embedder
from keras.models import load_model
import pickle

with open('users.pkl','rb') as file:
    cla=pickle.load(file)

#classifier trained on the output of facenet 128-d embedding 
model=load_model('face.h5')
cap=cv2.VideoCapture(0)
ref = True
detector = cv2.CascadeClassifier(
    '/home/akura/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
while ref:
    ref, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for i, rect in enumerate(rects):
        (x, y, w, h) = rect
        #face=frame[y:y+h][x:x+w]
        vect=embedder(frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        val=model.predict (np.expand_dims (np.array(vect),axis=0))
        test=np.argmax(val, axis=1)

        print(test[0],cla[1])
        cv2.putText(frame, cla[test[0]], (x, y),
                    cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Generating Data', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
