from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import render_to_string
from . models import Destination
from .temp import func

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB# BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
# Create your views here.

def index(request):
    return render(request, 'index.html',)

    
def predict(request):
    if request.method == 'GET':
        if 'par1' in request.GET:
            par1 = float(request.GET['par1'])
        else:
            par1 = ''
        
        if 'par2' in request.GET:
            par2 = float(request.GET['par2'])
        else:
            par2 = ''

        if 'par3' in request.GET:
            par3 = float(request.GET['par3'])
        else:
            par3 = ''

        if 'par4' in request.GET:
            par4 = float(request.GET['par4'])
        else:
            par4 = ''

        if 'par5' in request.GET:
            par5 = float(request.GET['par5'])
        else:
            par5 = ''

        if 'par6' in request.GET:
            par6 = float(request.GET['par6'])
        else:
            par6 = ''

        if 'par7' in request.GET:
            par7 = float(request.GET['par7'])
        else:
            par7 = ''

                
    
    #sum  = par1 + par2+par3+par4+par5+par6+par7
    

    #result = func(par1, par2, par3, par4, par5, par6, par7)


    data = pd.read_csv('winequalityN.csv')
    
    #clmns = ['fixed acidity','volatile acidity', 'citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH']
    
    data=data.drop(["type"],axis=1)
    data.fillna(data.mean(), inplace=True)
    bins = (2,6.5,10)
    group_names = ['Average','Premium']
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
    
    #input1 = [0, 1, 3, 2, 4, 5,]
    input1 = [] 

    y_expect = y_test
    input1.append(0)
    input1.append(0)
    input1.append(0)
    input1.append(0)
    input1.append(0)
    input1.append(0)
    input1.append(0)
    
    
    #input = 
    y_pred = gaus.predict([input1])
    #print([input])
    
    if par7 == '':
        y_pred = ''
    
    demo = listToString(y_pred) 
    y_pred = demo

    sum = 5
    result = {'par1': par1, 'par2': par2, 'par3':par3, 'par4':par4, 'par5':par5, 'par6':par6, 'par7':par7, 'y_pred':y_pred}

    
    #input = X_test.iloc[100]
    
    return render(request, 'index-1.html', result)
    #return HttpResponse(y_pred)

def listToString(s):  
    # initialize an empty string 
    str1 = " " 

    # return string   
    return (str1.join(s))

def foodandwine(request):
    return render(request, 'index-2.html')

def about(request):
    return render(request, 'index-3.html')

def contact(request):
    return render(request, 'index-4.html')        

