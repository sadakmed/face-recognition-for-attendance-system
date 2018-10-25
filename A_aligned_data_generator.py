import cv2 
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import os 
from A_func import rect_to_bb,shape_to_np,FACIAL_LANDMARKS_5_IDXS,FACIAL_LANDMARKS_IDXS
import dlib


def align(shape,frame):
        shape = shape_to_np ( shape )
        if (len ( shape ) == 68):
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        else:
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        centerleft=np.mean(leftEyePts,axis=0,dtype=np.int16)
        centerright=np.mean(rightEyePts,axis=0,dtype=np.int16)
        dx=centerright[0]-centerleft[0]
        dy=centerright[1]-centerleft[1]
        alfa=np.degrees(np.arctan2(dy,dx))-180
        eyesCenter = ((centerleft[0] + centerright[0]) // 2,(centerleft[1] + centerright[1]) // 2)
        M=cv2.getRotationMatrix2D(eyesCenter,alfa,1)
        return cv2.warpAffine ( frame, M, (frame.shape[0],frame.shape[1]) )

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('predictor_face_land.dat')
#detector = cv2.CascadeClassifier('/home/akura/opencv-master/data/haarcascades/haarcascade_eye.xml')
detectedName=input('who you want to detect?')
num=0
to=50
print(detectedName)
pathtoface='dataset/{}'.format(detectedName)
if not os.path.isdir(pathtoface):
    os.mkdir(pathtoface)
cap=cv2.VideoCapture(0)
desiredLeftEye=(0.35, 0.35)

while True and num<50:
    ref,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    rects=detector(gray,1) 
    for i, rec in enumerate(rects):
        (x,y,w,h)=rect_to_bb(rec)
        shape=predictor(gray ,rec)
        output=align(shape,frame)
    cv2.imwrite(os.path.join("{}/{}.jpg".format(pathtoface,num)),output[y:y+h,x:x+w])
    num+=1
    cv2.imshow('Generating Data',frame)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break 
cap.release()
cv2.destroyAllWindows()
