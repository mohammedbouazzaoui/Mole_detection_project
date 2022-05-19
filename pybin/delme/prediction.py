# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:15:43 2022

@author: bmadmin
"""
import pandas as pd

def predictprice(model,house_json):

    X=pd.DataFrame([house_json])
    result=model.predict(X)

    return result