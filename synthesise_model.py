# 
"""
Skin cancer lesion classification using the HAM10000 dataset
Autokeras to find the best model. 
Dataset link:
https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
Data description: 
https://arxiv.org/ftp/arxiv/papers/1803/1803.10417.pdf
The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

labels=['nv','mel','bkl','bcc','akiec','vas','df']
"""


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

skin_df = pd.read_csv('./backup/data/HAM10000/HAM10000_metadata.csv')


SIZE=32

# label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
LabelEncoder()
#print(list(le.classes_))
 
skin_df['label'] = le.transform(skin_df["dx"]) 
#print(skin_df.sample(10))


# Distribution of data into various classes 
from sklearn.utils import resample
print(skin_df['label'].value_counts())

#Balance data.
# Many ways to balance data... you can also try assigning weights during model.fit
#Separate each classes, resample, and combine back into single dataframe
df=[]
for i in range(7):
    df.append( skin_df[skin_df['label'] == i])

###############################################
from sklearn.utils import shuffle
# split 20% test 20% validation 60% train for every class
df_test=[]
df_validate=[]
df_train=[]
for i in range(7):
    # shuffle the dataframe
    df[i] = shuffle(df[i], random_state=42)
    #split df in 20% as final testset
    len_df=len(df[i])
    len_test=int(len_df * 0.2)
    len_validation=len_test
    #len_rest=len_df - (len_test + len_validation)
    df_test.append(df[i].head(len_test))
    df[i]=df[i].tail(len_df - len_test)
    df_validate.append(df[i].head(len_validation))
    df_train.append(df[i].tail(len_df -len_validation -len_test))
##########################################
print('@$$$$$$$$$',df_train[0].shape,df_validate[0].shape,df_test[0].shape)
#input()
##########------------------###################################################################
# resample train set classes to each 500 samples 
n_samples=1000 
df_train_balanced=[]
for i in range(7):
    df_train_balanced.append( resample(df_train[i], replace=True, n_samples=n_samples, random_state=42) )
print('@$$$-----$',df_train_balanced[0].shape,df_train[0].shape,df_validate[0].shape,df_test[0].shape)
#input()
print('#prepare train set')
# prepare train set
# #########-------------###########################
def prepare_set(df_set,skin_df = skin_df):
    #put all classes together
    df_set = pd.concat([df_set[0], df_set[1], df_set[2], df_set[3], df_set[4], df_set[5], df_set[6]])
    #read images using image ID from the CSV file
    image_path = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join('./backup/data/HAM10000/HAM10000_images_part_?', '*.jpg'))}
    #Define the path and add as a new column
    df_set['path'] = skin_df['image_id'].map(image_path.get)
    #Use the path to read images.
    df_set['image'] = df_set['path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE,SIZE))))
    #Convert dataframe column of images into numpy array
    X = np.asarray(df_set['image'].tolist())
    X = X/255. # Scale values to 0-1. You can also used standardscaler or other scaling methods.
    Y=df_set['label'] #Assign label values to Y
    Y_cat = to_categorical(Y, num_classes=7) #Convert to categorical -> multiclass classification 
    return X, Y_cat
    #########-------------------------######################################################
# prepare all sets to be used
X_train_balanced,Y_cat_train_balanced = prepare_set(df_train_balanced)
X_test,Y_cat_test = prepare_set(df_test)
X_validate,Y_cat_validate = prepare_set(df_validate)

print('@$$$--x---$',X_test.shape,X_train_balanced.shape,X_validate.shape,'@@', df_train_balanced[0].shape,df_train[0].shape,df_validate[0].shape,df_test[0].shape)
#input()

print('@@@@@@@',len(X_train_balanced),len(X_test),len(X_validate))
# for selecting the best model we will use the validation set and split this into a train/test
#
# start model selection
print('# start model selection')
x_train_auto, x_test_auto, y_train_auto, y_test_auto = train_test_split(X_validate, Y_cat_validate, test_size=0.98, random_state=42)

# Define classifier for autokeras. Here we check 25 different models, each model 25 epochs
# for reasons of PC performane I can only use  2/2 !
clf = ak.ImageClassifier(max_trials=2) #MaxTrials - max. number of keras models to try
clf.fit(x_train_auto, y_train_auto, epochs=2)

#Evaluate the classifier on test data
print('#Evaluate the classifier on test data ')

_, acc = clf.evaluate(x_test_auto, y_test_auto)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
print('# get the final best performing model ')

model = clf.export_model()
print(model.summary())

print('#Save the model')
'#Save the model'
model.save('./pybin/models/bestof2model2epoch')
##
