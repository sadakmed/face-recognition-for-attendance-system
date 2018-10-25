
from keras.models import load_model
import numpy as np
import os 
import cv2
from tqdm import tqdm
face_emb = load_model('facenet_keras.h5')
def embedder(img):
    img = cv2.resize(img, (160, 160))
    img = img.astype('float')/255.0
    img = np.expand_dims(img, axis=0)
    return face_emb.predict(img)[0]
def upload_data():
    x_data=[]
    y_data=[]
    people = os.listdir('dataset')
    per = 0
    for x in people:
        for i in tqdm(os.listdir('dataset/'+x)):
            img = cv2.imread('dataset'+'/'+x+'/'+i, 1)
            
            embs = embedder(img)
            x_data.append(embs)
            y_data.append(per)
        per += 1
  
    return x_data,y_data


