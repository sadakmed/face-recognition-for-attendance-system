from recoNet import recoNet
import os 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from A_func import data


def shuffle( img, label, sed=None):
    #     if sed is not None:
    #     np.random.seed(2)
    permit = np.random.permutation ( img.size )
    img = img[permit]
    label = label[permit]
    return img, label

def flow_from_dataframe(img_data_gen, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname ( path_col[0] )
  
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


batch=8

x,y=data()
x,y=shuffle(x,y)
y=to_categorical(y,np.max(y)+1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
genImg=ImageDataGenerator(rescale=1.0/255)
imgTrain=flow_from_dataframe(genImg,x_train,y_train,batch_size=batch,target_size=(160,160))
imgTest=flow_from_dataframe(genImg,x_test,y_test,batch_size=batch,target_size=(160,160))
person=os.listdir ('dataset')
model = recoNet(len(person))

model.compile(optimizer=adam(lr=0.001,decay=0.00001),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(imgTrain,steps_per_epoch=x_train.shape[0]//batch,epochs=50,validation_data=imgTest,validation_steps=y_test.shape[0]//batch)

