import math
import argparse
import cv2
import os
import dlib
from matplotlib import pyplot as plt
from A_func import rect_to_bb,shape_to_np
from align import FaceAligner
import imutils

images = os.scandir ( 'dataset/makavelli' )
imgs = iter ( images )
imgpath = next ( imgs ).path

img = cv2.imread ( imgpath )
img=img[200:2000][:]
image = cv2.resize ( img, (256, 256) )
gray = cv2.cvtColor ( image, cv2.COLOR_BGR2GRAY )

detector = dlib.get_frontal_face_detector ()
predictor = dlib.shape_predictor ( 'predictor_face_land.dat' )
fa = FaceAligner ( predictor, desiredFaceWidth=256 )


rects = detector ( gray, 1 )

# loop over the face detectionsdef shuffle(self, img, label, sed=None):
    #     if sed is not None:
    #     np.random.seed(2)
    permit = np.random.permutation ( img.size )
    img = img[permit]
    label = label[permit]
    return img, label

def flow_from_dataframe(self, img_data_gen, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname ( path_col[0] )
    print ( '## Ignore next message from keras, values are replaced anyways' )
    df_gen = img_data_gen.flow_from_directory ( base_dir,
                                                class_mode='sparse',
                                                **dflow_args )
    df_gen.filenames = path_col
    df_gen.classes = y_col
    df_gen.samples = path_col.shape[0]
    df_gen.n = path_col.shape[0]
    df_gen._set_index_array ()
    df_gen.directory = ''  # since we have the full path
    print ( 'Reinserting dataframe: {} images'.format ( path_col.shape[0] ) )
    return df_gen
def bring_data():
        img = np.array ( [] )
        label = np.array ( [] )
        for i, cl in enumerate ( self.cls[:self.clal] ):
            imgcl = np.array ( [x.path for x in os.scandir ( cl )] )
            if self.card == 'all' :
                card = imgcl.size
            labelcl = np.array ( [i for j in range ( imgcl.size )] )
            img = np.concatenate ( (img, imgcl[:card]), axis=0 )
            label = np.concatenate ( (label, labelcl[:card]), axis=0 )
        return img, label

for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb ( rect )
    faceOrig = imutils.resize ( image[y:y + h, x:x + w], width=256 )
    faceAligned = fa.align ( image, gray, rect )

'''for i,rect in enumerate(rects):
    shape=predictor(gray,rect)
    shape=shape_to_np(shape)
    (x,y,w,h)=rect_to_bb(rect)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(img,'face #{}'.format(i+1),(x-10,y-10),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,0.5,(0,255,0),2)
    for (x,y) in shape:
        cv2.circle(img,(x,y),1,(0,0,255),-1)
      '''

#plt.imshow ( faceOrig )
plt.imshow ( faceAligned )
plt.show ()

# print(img.shape)
