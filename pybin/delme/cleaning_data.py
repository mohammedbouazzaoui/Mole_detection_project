# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:13:08 2022

@author: bmadmin
"""

import pandas as pd
import numpy as np
from pybin.myfunctions import debug

DEBUG=False

def clean_immodata(inputfile:str = "./data/data_homes.csv",cleaned_file:str = "./data/data_homes_cleaned.csv"):
    debug(DEBUG,"clean_immodata")

    #
    # function : will clean the immoweb data containing all the houses
    #
    # input: file with immoweb data
    # output: clean data
    #

    df=pd.read_csv(inputfile,delimiter=",")
    
    #MAIN

    #inputfile = "./data/data_homes.csv"
    df=pd.read_csv(inputfile,delimiter=",")
    
    # Cleaning steps
    ################
    # Checking how many rows of each attribute are NaN
    #df.isna().sum()
    
    # drop columns
    dropcolmns=['classified.type','classified.condition.isNewlyBuilt','classified.transactionType','customer.id','classified.visualisationOption','classified.id','screen.language','screen.name','customer.networkInfo.id','customer.networkInfo.name',\
                'customer.groupInfo.name','customer.groupInfo.id','user.loginStatus',\
                'Unnamed: 0','user.personal.language','user.id',\
                'customer.name','customer.family']
    df = df.drop(columns=dropcolmns)
    df=df.drop_duplicates()
    debug(DEBUG,"2clean_immodata")
    # drop rows "fromprice - toprice" these are 'housegroups'
    df=df[df['classified.price'].str.find('-') == -1]
    # drop bad rows
    df=df[df['classified.kitchen.type'] != 'classified.kitchen.type']
    df=df[df['classified.building.condition'] != 'classified.building.condition']
    df=df[df['classified.price'] != 'no price']
    
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('not installed' ,1)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('installed' ,2)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('semi equipped' ,3)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('hyper equipped' ,4)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('usa uninstalled' ,1)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('usa installed' ,2)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('usa semi equipped' ,3)
    df['classified.kitchen.type'] = df['classified.kitchen.type'].replace('usa hyper equipped' ,4)
    
    df['classified.building.condition'] = df['classified.building.condition'].replace('to renovate',1)
    df['classified.building.condition'] = df['classified.building.condition'].replace('to restore',2)
    df['classified.building.condition'] = df['classified.building.condition'].replace('to be done up',3)
    df['classified.building.condition'] = df['classified.building.condition'].replace('good',5)
    df['classified.building.condition'] = df['classified.building.condition'].replace('just renovated',6)
    df['classified.building.condition'] = df['classified.building.condition'].replace('as new',7)
    debug(DEBUG,"3clean_immodata")
    # set 0
    ################
    #----------- replace nan/None
    flds=[
          "classified.outdoor.garden.surface",
          "classified.parking.parkingSpaceCount.indoor",
          "classified.parking.parkingSpaceCount.outdoor",
          "classified.bedroom.count"
         ]
    for fld in flds:
        df[fld]=df[fld].replace(np.nan, 0)
        df[fld]=df[fld].replace('None', 0)
    
    #----------- replace nan,False,True
    flds=[
          "classified.atticExists",
          "classified.basementExists",
          "classified.outdoor.terrace.exists",
          "classified.specificities.SME.office.exists",
          "classified.wellnessEquipment.hasSwimmingPool"
            ]
    for fld in flds:
        df[fld]=df[fld].replace(np.nan, 0)
        df[fld]=df[fld].replace('false', 0)
        df[fld]=df[fld].replace('true', 1)

    df=df.dropna()
    debug(DEBUG,"4clean_immodata")
    #change type
    df['classified.price'] = df['classified.price'].astype(float)
    df['classified.building.constructionYear'] = df['classified.building.constructionYear'].astype(float)
    df['classified.certificates.primaryEnergyConsumptionLevel'] = df['classified.certificates.primaryEnergyConsumptionLevel'].astype(float)
    df['classified.bedroom.count'] = df['classified.bedroom.count'].astype(np.int64)
    df['classified.land.surface'] = df['classified.land.surface'].astype(float)
    df['classified.outdoor.garden.surface'] = df['classified.outdoor.garden.surface'].astype(float)
    df['classified.parking.parkingSpaceCount.indoor'] = df['classified.parking.parkingSpaceCount.indoor'].astype(np.int64)
    df['classified.parking.parkingSpaceCount.outdoor'] = df['classified.parking.parkingSpaceCount.outdoor'].astype(np.int64)
    dum=pd.get_dummies(df['classified.energy.heatingType'])
    df=df.join(dum)
    df=df.drop(columns=['classified.energy.heatingType'])
    df=df.drop(columns=['classified.subtype'])
    debug(DEBUG,"5clean_immodata")

    df.to_csv("./data/data_homes_cleaned.csv",index=False) 

