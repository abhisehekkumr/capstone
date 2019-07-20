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



class AdaBoosting:
    
    def __init__(self,dataset,T,test_dataset,target):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
        self.target = target
    
    
    def underSampling(self):
        
        # return count of uniure values as a list and in out case because of 
        # binary classification it will return list haveing index -1 and 1 \
        # along with their counts and that count will ignore any null if exists
        classes_in_dataset = self.Y.value_counts()
        
        # toatl number of negative instances is stored at negative index
        # basically class name act as index :)
        
        negative_class = classes_in_dataset[-1]
        positive_class = classes_in_dataset[1]
        
        # empty list
        keep_list = []
        delete_list = []
        
        
        # dividing dataset into two parts
        # if we have majority in negative class delete kist will keeo instances
        # belonging to negative class and it positive class in majority than
        # delete list will contain postive class instances
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
            
        #removing instances from delete list to achieve balance ration
        # random.choice will give as random number from given range as range
        # is dynamically updated
        while(len(delete_list)  > len(keep_list)):
            index = random.choice(range(len(delete_list)))
            delete_list.pop(index)
        
        all_list = delete_list + keep_list
        return sorted(all_list, key = lambda x:x[2])
    
    
    def fit(self):

        X = self.dataset.drop([self.target],axis=1)
        Y = self.dataset[self.target].where(self.dataset[self.target]==1,-1)

        Evaluation = pd.DataFrame(Y.copy())
        Evaluation['weights'] = 1/len(self.dataset) 
        
        
        
        alphas = [] 
        models = []
        
        for t in range(self.T):
            
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1)    
            model = Tree_model.fit(X,Y,sample_weight=np.array(Evaluation['weights'])) 
            
            
           
            
            models.append(model)
            predictions = model.predict(X)
            
            result = np.where(predictions == Y, 0 , 1)
            err = np.sum(Evaluation['weights']*result)/np.sum(Evaluation['weights'])
            alpha = np.log((1-err)/err)*0.5
            alphas.append(alpha)
            Evaluation['weights'] *= np.exp(alpha*result)
            
           
            
           
        
        self.alphas = alphas
        self.models = models
            
    def predict(self):
        X_test = self.test_dataset.drop([self.target],axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset[self.target].reindex(range(len(self.test_dataset))).where(self.dataset[self.target]==1,-1)
    
       
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) 
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
           
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
   
        
        
######Plot the accuracy of the model against the number of stump-models used##########
number_of_base_learners = 50
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

dataset = pd.read_excel('HSR.xlsx')
dataset = dataset.iloc[:,1:]

names = dataset.columns
target = names[0]

for i in range(number_of_base_learners):
    model = AdaBoosting(dataset,i,dataset,target)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ',number_of_base_learners,'base models we receive an accuracy of ',model.accuracy[-1]*100,'%')    
                 
plt.show()        
