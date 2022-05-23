# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:17:15 2022

@author: bmadmin

import sys 
import os
sys.path.append(os.path.abspath("C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/mole_detection/"))

"""
############################

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

from tensorflow.keras.preprocessing.image import save_img

################################

from flask import Flask,render_template,request

import io
from pybin.mylib.myfunctions import debug







app = Flask(__name__, template_folder="./pybin/templates")

#global modelfile

DEBUG=True

#######################################################
debug(DEBUG,"start")
#load the model
#model=tf.keras.models.load_model('./cifar_model.h128_3')
#model=tf.keras.models.load_model('./pybin/models/cifar_model.h128_3')
global modelfile
modelfile="./pybin/models/activemodel"
debug(DEBUG,modelfile)
model=tf.keras.models.load_model(modelfile)
#debug(DEBUG,model.summary())

# load the image
#img1 = load_img('./pybin/images/ISIC_0029313.jpg')

######################################################

def predictimage(img,model):
    # report details about the image
    debug(DEBUG,type(img))
    debug(DEBUG,img.format)
    debug(DEBUG,img.mode)
    debug(DEBUG,img.size)
    # show the image
    #img.show()

    # convert to numpy array
    img = img_to_array(img)
    img = cv2.resize(img, (32, 32))

    #use same scaling as in training model ????????
    img=img/255
    
    img=np.reshape(img, (-1, 32, 32, 3)) 
    
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
    
    

        
@app.route('/info/', methods = ['POST', 'GET'])
def info():
    
    #only way to get the summary into a string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    info=summary_string.splitlines()
    debug(DEBUG,modelfile)

    return render_template('info.html',info=info)



@app.route('/image_show_prediction/', methods = ['POST', 'GET'])
def image_show_prediction():
    
    debug(DEBUG,"image_predict")
    form_data = request.form
    
    debug(DEBUG,form_data)
  
    return render_template('image_show_prediction.html')

@app.route('/image_load/', methods = ['POST', 'GET'])
def image_load():
    return render_template('image_load.html')
    
@app.route('/image_predict/', methods = ['POST', 'GET'])
def image_predict():

    debug(DEBUG,"file is1")
    
    form_data = request.form   # get image filename
    debug(DEBUG,form_data)
    imagename=form_data['myfile']
    imagefile="./pybin/images/"+imagename
    
    
    debug(DEBUG,imagefile)

  
    # load the image
    image = load_img(imagefile)
    save_img('./static/imgtopredict.jpg',image)
    debug(DEBUG,"after load @image")

    
    predict=predictimage(image,model)

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
    
    #print(seven[predict])
    debug(DEBUG,seven[predict])
    return render_template('image_show_prediction.html',predict=seven[predict],imagename=imagename)

@app.route('/camera/', methods = ['POST', 'GET'])
def camera():     
    #import cv2 as cv
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    if result:
        cv2.imwrite("./static/imgtopredict.jpg", image)
    else:
        print("No image detected. Please! try again")
    imagefile='./static/imgtopredict.jpg'
    image = load_img(imagefile)
    predict=predictimage(image,model)

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
    
    #print(seven[predict])
    debug(DEBUG,seven[predict])
    imagename='camera'
    return render_template('image_show_prediction.html',predict=seven[predict],imagename=imagename)
 

@app.route('/', methods = ['POST', 'GET'])
def roott():

    return render_template('image_load.html')  
#######################################
# MAIN
#######################################

#######################################
app.run(host='localhost', port=5000)
