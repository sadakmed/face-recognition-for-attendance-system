from  keras  import layers ,Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np 

def net(per):
    Input_shape=(128,)
    input_size=Input(Input_shape)
    x=layers.Dense(64)(input_size)
    x=layers.Activation('relu')(x)
    x=layers.Dense(32)(x)
    x=layers.Activation('relu')(x)
    x=layers.Dense(32)(x)
    x=layers.Activation('relu')(x)
    x=layers.Dense(per)(x)
    output=layers.Activation('softmax')(x)

    return Model(input_size,output)
model=net(2)
model.summary()







