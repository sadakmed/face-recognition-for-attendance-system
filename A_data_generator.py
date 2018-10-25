import os 
import cv2 
from A_func import rect_to_bb
num=0
cap=cv2.VideoCapture(0)
ref=True
detector=cv2.CascadeClassifier('/home/akura/opencv-master/data/haarcascades/haarcascade_frontalface_alt.xml')
while ref and num<60 :
    ref,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    
    for i,rect in enumerate(rects):
        (x,y,w,h)=rect
        #face=frame[y:y+h][x:x+w]
        num+=1
        cv2.imwrite(os.path.join('dataset/{}.jpg'.format(num)),frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('Generating Data',frame)



    if cv2.waitKey(1) & 0xFF == ord('q') :
        break 
cap.release()
cv2.destroyAllWindows()
