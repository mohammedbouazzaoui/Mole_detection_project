

import numpy as np
import pandas as pd 
import shutil
import keras as tf
from keras.preprocessing.image import ImageDataGenerator

''' 
def create_dir_structure():
    #run it once to create and prepare for augmenting
    seven={
                'nv':'Melanocytic nevi',
                'mel':'Melanoma',
                'bkl':'Benign keratosis-like lesions',
                'bcc':'Basal cell carcinoma',
                'akiec':'Actinic keratoses',
                'vas':'Vascular lesions',
                'df':'Dermatofibroma'
                }
    #create dir structure :
    # ./backup/data/HAM10000/HAM10000_metadata.csv    <-- list containing picsnames with prediction
    # ./backup/data/HAM10000/ALL          <-- put all available images here 
    # create for every class a directory as follows
    #
    # ./backup/data/HAM10000/myaugmenting/nv
    # ./backup/data/HAM10000/myaugmenting/mel
    # ./backup/data/HAM10000/myaugmenting/bkl
    # ./backup/data/HAM10000/myaugmenting/bcc
    # ./backup/data/HAM10000/myaugmenting/akiec
    # ./backup/data/HAM10000/myaugmenting/vas
    # ./backup/data/HAM10000/myaugmenting/df

    df=pd.read_csv("./backup/data/HAM10000/HAM10000_metadata.csv")

    df['frm']='./backup/data/HAM10000/ALL/'+df['image_id']+'.jpg'
    df['to']='./backup/data/HAM10000/myaugmenting/'+df['dx']+'/'+df['image_id']+'.jpg'

    df=df[['frm','to']]
    fromtolist=df.values.tolist()

    # copy all images to the corresponding directory
    i=0
    j=0
    for frm,to in fromtolist:
        try:
            shutil.copyfile(frm, to)
        except:
            j+=1
        i+=1
        print (f"\r>> copied: {i}    could not copy: {j}", end='', flush=True) 
''' 

#create augmented images
directory='./backup/data/HAM10000/picaugment/'

save_directory='./backup/data/HAM10000/picaugmented'
seven={
                'nv':'Melanocytic nevi',
                'mel':'Melanoma',
                'bkl':'Benign keratosis-like lesions',
                'bcc':'Basal cell carcinoma',
                'akiec':'Actinic keratoses',
                'vas':'Vascular lesions',
                'df':'Dermatofibroma'
                }
seven={
                'df':'Dermatofibroma'
                }
   
imagegenerator = ImageDataGenerator(
        rescale=1 / 255.0,
        rotation_range=20,
        zoom_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.20)
for classname in seven.keys():
    print(classname)
    save_directory='./backup/data/HAM10000/picaugment/' + classname + '_out'
    directory='./backup/data/HAM10000/picaugment/' + classname + '/'

    img_gen=imagegenerator.flow_from_directory(
        directory=directory,
        target_size=(256, 256),
        color_mode='rgb',
        classes=[classname],
        class_mode='categorical',
        batch_size=1,
        shuffle=True,
        seed=None,
        save_to_dir=save_directory,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    sevennmbr={
                'nv':6705,
                'mel':1113,
                'bkl':1099,
                'bcc':514,
                'akiec':327,
                'vas':142,
                'df':115
                }
    #nmbr=[6705,1113,1099,514,327,142,115]
    i=0
    for img in img_gen:
        i+=1
        print (f"\r>> img created: {i}", end='', flush=True)
        print(type(img))
        if i == 500:
            break

