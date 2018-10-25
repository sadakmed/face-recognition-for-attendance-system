import cv2
import numpy as np
from A_func import data , bring_data
import os
from keras.utils import to_categorical 
i=2
x,y=data()

y=to_categorical(y,np.max(y)+1)
    
print(y)