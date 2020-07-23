import cv2
import numpy as np
from PIL import Image
import os
import numpy as np
import cv2
import os
import h5py
import dlib
# from imutils import face_utils
from keras.models import load_model
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout,BatchNormalization
from keras.layers import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras import backend as K 
from sklearn.model_selection import train_test_split




def model(input_shape,num_classes):
   
      # Build the network model
    model = Sequential()

    model.add(Conv2D(6, (5, 5), padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(16, (5, 5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
    model.summary()
    return model
    
