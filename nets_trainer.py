import pickle
from A_embedder import upload_data,embedder
#from A_classNet import net
from keras.optimizers import adam
from sklearn import svm 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
from tqdm import tqdm
import cv2 
import numpy as np


data={}
x_data = []
y_data = []
users={}
people = os.listdir('dataset')
per = 0
for x in people:
        users[per]=x
        j=0
        for i in os.listdir('dataset/'+x):
            img = cv2.imread('dataset'+'/'+x+'/'+i, 1)
            embs = embedder(img)
            x_data.append(embs)
            y_data.append(per)
            data[per]=embs
            
        per += 1

x_data=np.array(x_data)
print(len(y_data))
y_data = np.array(y_data).reshape(len(y_data),)
y_data=to_categorical(y_data,per+1)
x_tr, x_test,y_tr,y_test = train_test_split(x_data, y_data, test_size=0.1,random_state=5)

model=net(per+1)
model.compile(optimizer=adam(lr=0.001,decay=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_tr,y_tr,epochs=100,validation_data=(x_test,y_test),batch_size=8,shuffle=True)
model.save('face.h5')


with open('users.pkl','wb') as file:
    pickle.dump(users,file)
    



