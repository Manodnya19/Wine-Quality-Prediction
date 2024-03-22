# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random as rd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB# BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler

input1 = []

def func(par1, par2, par3, par4, par5, par6, par7):
    data = pd.read_csv('winequalityN.csv')
    
    #clmns = ['fixed acidity','volatile acidity', 'citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH']
    
    data=data.drop(["type"],axis=1)
    data.fillna(data.mean(), inplace=True)
    bins = (2,5,10)
    group_names = ['average','premium']
    data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
    data['quality'].unique()
    used_features =[
            "quality",
        "fixed acidity",
        "citric acid",
        "chlorides",
        "residual sugar",
    ]
    
    X = data.drop(used_features,axis = 1)
    y = data['quality']
    
    #sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=17)
    
    #gnb = GaussianNB(priors = None
    #MultiNB = MultinomialNB()
    #MultiNB.fit(X_train,y_train)
    #print(MultiNB)
    #y_expect = y_test
    #y_pred = MultiNB.predict(X_test)
    #print( accuracy_score(y_expect,y_pred))
    
    #2
    gaus = GaussianNB(priors = None)
    gaus.fit(X_train,y_train)
    print(gaus)
    
    y_expect = y_test
    """
    input1[1] = par1
    input1[2] = par2
    input1[3] = par3
    input1[4] = par4
    input1[5] = par5
    input1[6] = par6
    input1[7] = par7
    """ 
    input1.append(par1)
    input1.append(par2)
    input1.append(par3)
    input1.append(par4)
    input1.append(par5)
    input1.append(par6)
    input1.append(par7)
    
    #input = X_test.iloc[100]
    y_pred = gaus.predict([input1])
    print(y_pred)
    return y_pred


#print( accuracy_score(y_expect,y_pred))


#gaus =gaus(priors=None, var_smoothing=1e-09)

#TRAINING



# Print results
#print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
 #     .format(
  #        X_test.shape[0],
# =============================================================================
#           (X_test["alcohol"] != y_pred).sum(),
# =============================================================================
   #       100*(1-(X_test["alcohol"] != y_pred).sum()/X_test.shape[0])
#))

