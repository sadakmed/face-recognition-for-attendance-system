import numpy as np 
from keras.models import load_model,Model
from keras import Sequential , layers 
from keras.layers import add

def recoNet(person):
    
    faceNet =load_model('facenet_keras.h5')

    for layer in faceNet.layers:
        layer.trainable =False
    
    x=faceNet.output
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.5)(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=0.5)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.5)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.5)(x)
    x = layers.Dense(person)(x)
    output = layers.Activation('softmax')(x)
    myModel=Model(inputs=faceNet.input,outputs=output)
    myModel.summary()

   
    return myModel




