# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:17:15 2022

This program is the MVP for the 'Mole detection' project
########################################################
@author: Bouazzaoui Mohammed

"""

import os
import io

import numpy as np
import pandas as pd

from glob import glob
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical

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

from flask import Flask, render_template, request

from pybin.mylib.myfunctions import debug

np.random.seed(42)

app = Flask(__name__, template_folder="./pybin/templates")

# set DEBUG to 'True' to output debug information
DEBUG = False

global modelfile

# get the active model
modelfile = "./pybin/models/activemodel"
debug(DEBUG, modelfile)
model = tf.keras.models.load_model(modelfile)
debug(DEBUG, modelfile)


def predictimage(img, model):
    # Function will return a prediction
    #
    # Input : image, model
    # Return : prediction string
    #

    # report details about the image
    debug(DEBUG, type(img))
    debug(DEBUG, img.format)
    debug(DEBUG, img.mode)
    debug(DEBUG, img.size)

    # convert to numpy array
    img = img_to_array(img)
    img = cv2.resize(img, (64, 64))

    # new image has to be rescaled/reshaped as in training model 
    img = img / 255.0
    img = np.reshape(img, (-1, 64, 64, 3))

    # predict
    result = model.predict(img)
    result = result[0]
    debug(DEBUG, result)

    # The 7 classes of skin cancer lesions are:
    seven = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vas": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    le = LabelEncoder()
    le.fit(list(seven.keys()))

    # transform result to %
    som = sum(result)
    reslist = []
    for i in range(7):
        reslist.append([int(100 * result[i] / som), le.classes_[i]])
    reslist.sort()
    debug(DEBUG, reslist)

    # set '> 57%'  as a treshold for a good prediction
    if reslist[-1][0] > 57:
        return f"Predicted type : {seven[reslist[-1][1]]}  with {reslist[-1][0]}% accuratie."
    else:
        return f"Cannot determine type :    # {reslist[-1][1]} {reslist[-1][0]}% \
                                            # {reslist[-2][1]} {reslist[-2][0]}% \
                                            # {reslist[-3][1]} {reslist[-3][0]}% \
                                            # {reslist[-4][1]} {reslist[-4][0]}% \
                                            # {reslist[-5][1]} {reslist[-5][0]}% \
                                            # {reslist[-6][1]} {reslist[-6][0]}% \
                                            # {reslist[-7][1]} {reslist[-7][0]}% \
                                            "


@app.route("/info/", methods=["POST", "GET"])
def info():
    # Function will return the model information
    #
    # Input :  
    # Return : render_template 
    
    # get the summary into a string
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_string = stream.getvalue()
    stream.close()

    info = summary_string.splitlines()
    debug(DEBUG, modelfile)

    return render_template("info.html", info=info)


@app.route("/image_show_prediction/", methods=["POST", "GET"])
def image_show_prediction():
    # Function  
    #
    # Input :  
    # Return : render_template  
    #
    debug(DEBUG, "image_predict")
    form_data = request.form

    debug(DEBUG, form_data)

    return render_template("image_show_prediction.html")


@app.route("/image_load/", methods=["POST", "GET"])
def image_load():
    # Function  
    #
    # Input :  
    # Return : render_template 
    #
    return render_template("image_load.html")


@app.route("/image_predict/", methods=["POST", "GET"])
def image_predict():
    # Function does prediction
    #
    # Input :  
    # Return : render_template 
    #
 
    form_data = request.form  # get image filename
    debug(DEBUG, form_data)
    imagename = form_data["myfile"]
    if imagename == "":
        return render_template("image_load.html")

    imagefile = "./pybin/images/" + imagename

    debug(DEBUG, imagefile)

    # load the image
    image = load_img(imagefile)
    save_img("./static/imgtopredict.jpg", image)

    predict = predictimage(image, model)

    # The 7 classes of skin cancer lesions included in this dataset are:
    seven = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vas": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    debug(DEBUG, predict)
    return render_template(
        "image_show_prediction.html", predict=predict, imagename=imagename
    )


@app.route("/camera/", methods=["POST", "GET"])
def camera():
    # Function does take an image and predicts
    #
    # Input :  
    # Return : render_template 
    #

    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
    result, image = cam.read()
    if result:
        cv2.imwrite("./static/imgtopredict.jpg", image)
    else:
        print("No image detected. Please! try again")
    imagefile = "./static/imgtopredict.jpg"
    image = load_img(imagefile)
    predict = predictimage(image, model)

    # The 7 classes of skin cancer lesions included in this dataset are:
    seven = {
        "nv": "Melanocytic nevi",
        "mel": "Melanoma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vas": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    # print(seven[predict])
    debug(DEBUG, predict)
    imagename = "Camera"
    return render_template(
        "image_show_prediction.html", predict=predict, imagename=imagename
    )


@app.route("/", methods=["POST", "GET"])
def roott():
    # Function : dummy just show main screen
    #
    # Input :  
    # Return : render_template 
    #
    return render_template("image_load.html")


#######################################
# Start the local webserver
#######################################

app.run(host="localhost", port=5000)
