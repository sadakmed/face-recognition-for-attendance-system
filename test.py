import os 
import cv2
import dlib
from A_func import rect_to_bb,shape_to_np
#vid=VideoStream(src=0).start()
cap=cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('/home/akura/opencv-master/data/haarcascades/haarcascade_frontalface_alt2.xml')
detectorDlib=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('predictor_face_land.dat')

total=0
if not os.path.isdir('newData') :
    os.mkdir('newData')

while True:
    ref , frame=cap.read()
    orig=frame.copy()
   # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector.detectMultiScale(cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA),scaleFactor=1.1,minNeighbors=5,minSize=(30,30))
   # rects=detectorDlib(gray,1)
    for i,rec in enumerate(rects):
        (x,y,w,h)=rec
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',frame) 
    key=cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('k') :
        p=os.path.sep.join(['dataset/makavelli',"{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p,orig)
        total+=1
       
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()