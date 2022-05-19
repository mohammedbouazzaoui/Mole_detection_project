# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 13:28:54 2022

@author: bmadmin
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from pybin.myfunctions import debug  

DEBUG = False
        
class Inputdata:
    columns=[]
    def __init__(self,datafile: str):
        self.datafile=datafile
        self.yname='classified.price'
        self.Xy=[]
        self.X=[]
        self.y=[]
        self.X_train=[]
        self.X_test=[]
        self.y_train=[]
        self.y_test = []
        self.random_state = 42
        self.train_size = 0.80
        self.col_tokeep=[]
        
    def colmns(self,col_tokeep=['classified.price',
                                'classified.zip', 
                                'classified.building.constructionYear',
                                'classified.bedroom.count',
                                'classified.outdoor.garden.surface',
                                'classified.building.condition']):
        debug(DEBUG,col_tokeep)
        self.col_tokeep=col_tokeep
        
    def prepare(self):
        self.colmns()
        self.read_csv()
        self.split_data()
           
    def read_csv(self):
        self.Xy=pd.read_csv(self.datafile)
        debug(DEBUG,self.Xy.columns)
        self.Xy=self.Xy[self.col_tokeep]
        z=list(self.Xy.columns)
        z.remove('classified.price')
        Inputdata.columns=z
        
    def split_data(self):

        self.X=self.Xy.drop(columns=[self.yname])
        self.y=self.Xy[self.yname]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, random_state=self.random_state)

    def balance_data(self):
        pass    
    
class Model():
    model_storage={}   
    def __init__(self, filename: str, model = []):
        self.model = model
        self.filename = filename
        self.regressor=[]
        self.accuracy_score =[]
        mname="model"+str(len(Model.model_storage))
        Model.model_storage[mname]=self
        self.columns=[]
        
    def fit_model(self,X_train,y_train,X_test,y_test): #train the model
        self.regressor=self.model.fit(X_train,y_train)
        self.accuracy_score=self.regressor.score(X_test,y_test)
        self.columns=Inputdata.columns

    
    def save(self):
        pickle.dump(self.model, open(self.filename, 'wb'))
        
    def load(self):
        # Function : loads a model file
        # load the model from disk
        self.model = pickle.load(open(self.filename, 'rb'))
        
    def predict_model(self,X):
        ypred=self.model.predict(X)
        return ypred       
        
    def test_model():
        pass
    def visu_model():
        pass   