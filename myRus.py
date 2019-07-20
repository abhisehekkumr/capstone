#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:55:20 2019

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
from sklearn.svm import SVC
import random
from numpy import array
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

class AdaBoosting:
    
    def __init__(self,dataset,Y,T,test_dataset,target):
        self.X = dataset
        self.Y = Y
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
        self.target = target
        self.weights = []
        
        for i in range(len(self.Y)):
            self.weights.append(1/len(self.Y))
    
    
    def underSampling(self):
        
        classes_in_dataset = self.Y.value_counts()
        negative_class = classes_in_dataset[-1]
        positive_class = classes_in_dataset[1]
        
        keep_list = []
        delete_list = []
        
        if positive_class > negative_class:
            
            for i in range(len(self.Y)):
                
                if self.Y[i] == 1:
                    delete_data = [i, self.X.iloc[i,:], 1]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X.iloc[i,:], -1]
                    keep_list.append(keep_data)
        else:
            
            
            for i in range(len(self.Y)):
                
                if self.Y[i] == -1:
                    delete_data = [i, self.X.iloc[i,:], -1]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X.iloc[i,:], 1]
                    keep_list.append(keep_data)
            
       
        while(len(delete_list)  > len(keep_list)):
            index = random.choice(range(len(delete_list)))
            delete_list.pop(index)
        
        all_list = delete_list + keep_list
        return sorted(all_list, key = lambda x:x[2])
    
    
    def fit(self):


        alphas = [] 
        models = []
        print(self.T)
        for t in range(1,self.T + 1):
            #print(t)
            subset = self.underSampling()
            
            dataset = [s[1] for s in subset]
            index = [s[0] for s in subset]
            Y = [s[2] for s in subset]
            
            weights = []
            for i in range(len(dataset)):
                weights.append(self.weights[index[i]])
                
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)    
            model = Tree_model.fit(dataset,Y,sample_weight=np.array(weights)) 
            
            models.append(model)
            predictions = model.predict(self.X)
            
            result = np.where(predictions == self.Y, 0, 1)
            #print('result is ',result)
            err = np.sum(self.weights*result)/np.sum(self.weights)
            
            alpha = np.log((1-err)/err)*0.5
            alphas.append(alpha)
            #print(index)
            sum = 0
            for i in range(len(self.weights)):
                self.weights[i] *= np.exp(alpha*result[i])
                sum += self.weights[i]
            
            for i in range(len(weights)):
                self.weights[i] = self.weights[i]/sum
            
            # svm independent from sample but with class
            # 93.46
            classifier = SVC(kernel = 'rbf', gamma = 'auto', class_weight = 'balanced')
            classifier.fit(dataset,Y)
            models.append(classifier)
            alphas.append(1)
            
            cluster = Birch(branching_factor = 50, n_clusters=2, threshold=0.5,compute_labels=True)
            cluster.fit(dataset)
            predictions = cluster.predict(self.X)
            
            result = np.where(predictions == self.Y, 0, 1)
            #print('result is ',result)
            err = np.sum(self.weights*result)/np.sum(self.weights)
            
            alpha = np.log((1-err)/err)*0.5
            alphas.append(alpha)
            #print(index)
            sum = 0
            for i in range(len(self.weights)):
                self.weights[i] *= np.exp(alpha*result[i])
                sum += self.weights[i]
            
            for i in range(len(weights)):
                self.weights[i] = self.weights[i]/sum
            
            models.append(cluster)
            
            
        self.alphas = alphas
        self.models = models
            
    def predict(self):
        X_test = self.test_dataset
        Y_test = self.Y
    
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) 
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
           
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
   
        
        
######Plot the accuracy of the model against the number of stump-models used##########
number_of_base_learners = 5
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
dataset = pd.read_excel('HSR.xlsx')
dataset = dataset.iloc[:,1:]

names = dataset.columns
target = names[0]
Y = dataset[target].copy()
dataset = dataset.drop(target, axis = 1)

names = dataset.columns
from sklearn.decomposition import PCA
pca = PCA(n_components = 15)
components = pca.fit_transform(dataset)
dataset = pd.DataFrame(data = components)

for i in range(1, number_of_base_learners + 1):
    model = AdaBoosting(dataset,Y,i,dataset,target)
    model.fit()
    model.predict()
    if(len(model.accuracy) > 0):
        print('accuracy : ', model.accuracy[-1]*100)
ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ',number_of_base_learners,'base models we receive an accuracy of ',model.accuracy[-1]*100,'%')                   
plt.show()     
