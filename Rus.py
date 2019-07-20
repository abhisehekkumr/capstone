#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:30:00 2019

@author: abhi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as sps
import random
from sklearn.svm import SVC


class RusBoost:
    
    def __init__(self, train_instances, output_lables, n_classifier, test_data, target_col):
        self.X = train_instances
        self.Y = output_lables
        self.weights = []
        self.T = n_classifier
        self.models = []
        self.alphas = []
        self.accuracy = []
        self.predictions = None
        self.test_dataset = test_data
        self.target = target_col
        
        for i in range(len(self.X)):
            self.weights.append(1/len(self.X))
    
    def NormalizeOutput(self):
        self.Y = np.where(Y == 1, 1, -1)
        
        
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
        
        
        local_alphas = []
        local_models = []
        
        for i in range(self.T):
            
            print(i)
            
            sampled = self.underSampling()
            
            subset_weights = [self.weights[row[0]] for row in sampled]
            subset_X = [row[1] for row in sampled]
            subset_Y = [row[2] for row in sampled]
            
            Evaluation = pd.DataFrame(subset_Y)
            Evaluation['weights'] = subset_weights
            
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1) 
            model = Tree_model.fit(subset_X,subset_Y,sample_weight = np.array(Evaluation['weights']))
            
            local_models.append(model)
            predictions = model.predict(subset_X)
            #score = model.score(X,Y)
            
            
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == subset_Y,1,0) 
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != subset_Y, 1,0)
            
            #accuracy = sum(Evaluation['evaluation'])/len(Evaluation['evaluation'])
            error = np.sum(Evaluation['weights']*Evaluation['misclassified'])/np.sum(Evaluation['weights'])
            alpha = np.log((1-error)/error)*0.5
            local_alphas.append(alpha)
            #local_alphas.append(1)
            
            result = model.predict(self.X)
            for i  in range(len(self.Y)):
                
                if self.Y[i] != result[i]:
                    self.weights[i] *= np.exp(alpha*result[i])
            
            sum_weights = sum(self.weights)
            self.weights = [w/sum_weights for w in self.weights]
            
            Evaluation['weights'] *= np.exp(alpha*Evaluation['misclassified'])
            
            classifier = SVC(kernel = 'rbf', gamma = 'auto')
            classifier.fit(subset_X,subset_Y)
            local_models.append(classifier)
            local_alphas.append(1)
            
            '''
            result = classifier.predict(self.X)
            for i  in range(len(self.Y)):
                
                if self.Y[i] != result[i]:
                    self.weights[i] *= np.exp(alpha*result[i])
            
            sum_weights = sum(self.weights)
            self.weights = [w/sum_weights for w in self.weights]
            '''
            
        self.alphas = local_alphas
        self.models = local_models
        
    def predict(self):
        X_test = self.test_dataset.drop([self.target],axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset[self.target].reindex(range(len(self.test_dataset))).where(self.test_dataset[self.target]==1,-1)
    
       
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) 
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
           
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
        

######Plot the accuracy of the model against the number of stump-models used##########
number_of_base_learners = 10
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

dataset = pd.read_excel('HSR.xlsx')
dataset = dataset.iloc[:,1:]

names = dataset.columns
target = names[0]
X = dataset.drop(target, axis = 1)
Y = dataset[target].copy()


for i in range(number_of_base_learners):
    model = RusBoost(X,Y,i,dataset,target)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ',number_of_base_learners,'base models we receive an accuracy of ',model.accuracy[-1]*100,'%')    
                 
plt.show()        
