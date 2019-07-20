#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:38:59 2019

@author: abhi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps
import random



class RusBoost:
    
    def __init__(self, train_set, output_col, test_set, no_of_models, learning_rate):
        
        self.X = train_set
        self.Y = output_col
        self.test = test_set
        self.models = []
        self.alphas = []
        self.T = no_of_models
        self.accuracy = []
        self.weights = []
        self.rate = learning_rate
        self.accuracy = []
        
        for i in range(len(self.X)):
            self.weights.append(1/len(self.X))
            
            
        
    def random_sample(self):
        
            
            n_classes = self.Y.value_counts()
            negative = n_classes[-1]
            positive = n_classes[1]
            
            keep_list = []
            delete_list = []
            
            if positive > negative:        
                for i in range(len(self.Y)):
                    if self.Y[i] == 1:
                        delete_data = [ i , self.X.iloc[i,:], 1]
                        delete_list.append(delete_data)
                    else:
                        keep_data = [i , self.X.iloc[i,:], -1]
                        keep_list.append(keep_data)
            
            else:
                
                for i in range(len(self.Y)):
                    if self.Y[i] == -1:
                        delete_data = [i, self.X.iloc[i, : ], -1]
                        delete_list.append(delete_data)
                    else:
                        keep_data = [i, self.X.iloc[i, : ], 1]
                        keep_list.append(keep_data)
            
            
            while len(delete_list)  > self.rate*(len(delete_list + keep_list)):
                k = random.choice(range(len(delete_list)))
                delete_list.pop(k)
            
            all_list = delete_list + keep_list
            return sorted(all_list, key=lambda x:x[2])
            
            
            
    def fit(self):
        
        models = []
        alphas = []
        print(self.T)
        for t in range(self.T):
            subset = self.random_sample()
            
            local_weights = [self.weights[s[0]] for s in subset]
            local_dataset = [s[1] for s in subset]
            local_Y = [s[2] for s in subset]
            
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)
            model = Tree_model.fit(local_dataset,local_Y,sample_weight=np.array(local_weights)) 
            
            predictions = model.predict(self.X)
            
            error = 0
            
            for i in range(len(self.Y)):
                error += self.weights[i]*(1 - self.Y[i] + predictions[i])
                
            result = np.where(self.Y == predictions, 0,1)
            alpha = (error/(1 - error))
            
            for i in range(len(self.Y)):
                exponent = 0.5*(1 + self.Y[i] - result[i])
                self.weights[i] = self.weights[i]*np.power(alpha, exponent)
                
            
            sum = np.sum(self.weights)
            self.weights = [w/sum for w in self.weights]
            
            models.append(model)
            alphas.append(alpha)
            
        self.models = models
        self.alphas = alphas
        
    
    def predict1(self):
        X_test = self.test
        
        positive = 0
        negative = 0
        
        predictions = []
        pred = []
        
        for i in range(self.T):
            pred.append(self.models[i].predict(X_test))
        
        for i in range(len(self.Y)):
            
            for j in range(self.T):
                
                if pred[j][i] == 1:
                    positive += np.log(1/self.alphas[j])
                else:
                    negative += np.log(1/self.alphas[j])
            
            if positive >= negative:
                predictions.append(1)
            else:
                predictions.append(-1)
        
        return predictions
    
    def predict(self):
        X_test = self.test
        Y_test = self.Y
    
       
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) 
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
           
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
            
        
        

dataset = pd.read_excel('HSR.xlsx')
dataset = dataset.iloc[:,1:]
output = dataset.columns[0]
Y = dataset[output].copy()
dataset = dataset.drop([output], axis = 1)

rusboost = RusBoost(dataset, Y, dataset, 10, 0.5)
rusboost.fit()
pred = rusboost.predict()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y, pred)

