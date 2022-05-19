# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:17:15 2022

@author: bmadmin

import sys 
import os
sys.path.append(os.path.abspath("C:/Users/bmadmin/Desktop/Octocat/mohammedbouazzaoui/mole_detection/"))

"""

import numpy as np
from flask import Flask,render_template,request
from pybin.myfunctions import debug
import cv2
import tensorflow as tf
# example of loading an image with the Keras API
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.preprocessing.image import array_to_img
tf.keras.preprocessing.image.img_to_array

from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder="./pybin/templates")


DEBUG=True

#######################################################

#load the model
#model=tf.keras.models.load_model('./cifar_model.h128_3')
model=tf.keras.models.load_model('./pybin/models/cifar_model.h128_3')
print(model.summary())
# load the image
#img1 = load_img('./pybin/images/ISIC_0029313.jpg')

######################################################

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
    
    

        
@app.route('/info/', methods = ['POST', 'GET'])
def info():
    #global loadedmodel
    return render_template('info.html')



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
    imagefile=form_data['myfile']
    imagefile="./pybin/images/"+imagefile
    debug(DEBUG,"file is2")
    debug(DEBUG,imagefile)

    debug(DEBUG,"load image")
    # load the image
    image = load_img(imagefile)
    debug(DEBUG,"after load @image")
    
    
    ##############################################
    #image = load_img('./pybin/images/ISIC_0029313.jpg')
    
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
    
    print(seven[predict])
    debug(DEBUG,seven[predict])
    return render_template('image_show_prediction.html',predict=predict,imagefile=imagefile)


@app.route('/', methods = ['POST', 'GET'])
def root():

    return render_template('image_load.html')  
#######################################
# MAIN
#######################################

app.run(host='localhost', port=5000)
