# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Predict the class of your picture.
# prediction = new_model.predict(...
## SECOND TRIAL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image


np.random.seed(42)
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import autokeras as ak

import tensorflow as tf
from tensorflow import keras
import cv2

# example of loading an image with the Keras API
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img



#load the model
#model=tf.keras.models.load_model('./cifar_model.h128_3')
model=tf.keras.models.load_model('./pybin/models/cifar_model.h128_3')
print(model.summary())
# load the image
img1 = load_img('./pybin/images/ISIC_0029313.jpg')


def predictimage(img,model):
    # report details about the image
    print(type(img))
    print(img.format)
    print(img.mode)
    print(img.size)
    # show the image
    img.show()

    # convert to numpy array
    img = img_to_array(img)
    img = cv2.resize(img, (128, 128))
    img=np.reshape(img, (-1, 128, 128, 3)) 

    #predict
    result=model.predict(img)
    result=result[0]
    
    # label encoding to numeric values from text
    #The 7 classes of skin cancer lesions included in this dataset are:
    seven={
            'nv':'Melanocytic nevi',
            'mel':'Melanoma',
            'bkl':'Benign keratosis-like lesions',
            'bcc':'Basal cell carcinoma',
            'akiec':'Actinic keratoses',
            'vas':'Vascular lesions',
            'df':'Dermatofibroma'
            }
    le = LabelEncoder()
     
    le.fit(list(seven.keys()))
    LabelEncoder()
    print(le.classes_)
    for i in range(7):
        if result[i] == 1:
            break
    return str(le.classes_[i])
    
    
    
res=predictimage(img1,model)
print(res)

# label encoding to numeric values from text
#le = LabelEncoder()
#le.fit(skin_df['dx'])
#LabelEncoder()
#print(list(le.classes_))
 
#skin_df['label'] = le.transform(skin_df["dx"]) 
#print(skin_df.sample(10))

#The 7 classes of skin cancer lesions included in this dataset are:
seven={
        'nv':'Melanocytic nevi',
        'mel':'Melanoma',
        'bkl':'Benign keratosis-like lesions',
        'bcc':'Basal cell carcinoma',
        'akiec':'Actinic keratoses',
        'vas':'Vascular lesions',
        'df':'Dermatofibroma'
        }

print(seven[res])