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

class RandomSample:
    
    def __init__(self, dataset, output_col):
        self.dataset = dataset
        self.final_dataset = None
        self.subset = None
        self.output_col = output_col
        
    def get_subset(self):
        
        positive_class = self.dataset[self.dataset[self.output_col] == 1]
        negative_class = self.dataset[self.dataset[ self.output_col] == -1]
        
        
        if(len(positive_class) > len(negative_class)):
            self.subset = positive_class.sample(n = len(negative_class))
            self.final_dataset = pd.concat([self.subset, negative_class], ignore_index = True)
        else:
            self.subset = negative_class.sample(n = len(positive_class))
            self.final_dataset = pd.concat([self.subset, positive_class], ignore_index = True)
        
        self.final_dataset = self.final_dataset.sample( frac = 1 )
        return self.final_dataset


class Boosting:
    
    def __init__(self,dataset,T,test_dataset,target):
        self.dataset = dataset
        self.T = T
        self.test_dataset = test_dataset
        self.alphas = None
        self.models = None
        self.accuracy = []
        self.predictions = None
        self.target = target
        
    
    def fit(self):
        
        # Set the descriptive features and the target feature
        #X = self.dataset.drop([self.target],axis=1)
        #Y = self.dataset[self.target].where(self.dataset[self.target]==1,-1)
        # Initialize the weights of each sample with wi = 1/N and create a dataframe in which the evaluation is computed
        #Evaluation = pd.DataFrame(Y.copy())
        #Evaluation['weights'] = 1/len(self.dataset) # Set the initial weights w = 1/N
        self.dataset['weights'] = 1/len(self.dataset)
        self.dataset['locations'] = np.arange(len(dataset))
        # Run the boosting algorithm by creating T "weighted models"
        
       
        alphas = [] 
        models = []
        
        print(self.T)
        for t in range(self.T):
            # Train the Decision Stump(s)
            randomSample = RandomSample(dataset, self.target )
            subset = randomSample.get_subset()
            subset.sort_values('locations', axis = 0, ascending = True, inplace = True, na_position = 'last')
           
            locations = list(subset['locations'])
           
            
            X = subset.drop([self.target],axis=1)
            Y = subset[self.target].where(subset[self.target]==1,-1)
            
            Evaluation = pd.DataFrame(Y.copy())
            Evaluation = pd.DataFrame(subset['weights'])
            Evaluation[self.target] = subset[self.target]
            
            subset = subset.drop(['weights','locations'], axis = 1)
            
            
            Tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=1) # Mind the deth one --> Decision Stump
            
            # We know that we must train our decision stumps on weighted datasets where the weights depend on the results of
            # the previous decision stumps. To accomplish that, we use the 'weights' column of the above created 
            # 'evaluation dataframe' together with the sample_weight parameter of the fit method.
            # The documentation for the sample_weights parameter sais: "[...] If None, then samples are equally weighted."
            # Consequently, if NOT None, then the samples are NOT equally weighted and therewith we create a WEIGHTED dataset 
            # which is exactly what we want to have.
            model = Tree_model.fit(X,Y,sample_weight=np.array(Evaluation['weights'])) 
            
            # Append the single weak classifiers to a list which is later on used to make the 
            # weighted decision
            models.append(model)
            predictions = model.predict(X)
            score = model.score(X,Y)
            # Add values to the Evaluation DataFrame
            
            Evaluation['predictions'] = predictions
            Evaluation['evaluation'] = np.where(Evaluation['predictions'] == Evaluation[self.target],1,0)
            Evaluation['misclassified'] = np.where(Evaluation['predictions'] != Evaluation[self.target],1,0)
            # Calculate the misclassification rate and accuracy
            accuracy = sum(Evaluation['evaluation'])/len(Evaluation['evaluation'])
            misclassification = sum(Evaluation['misclassified'])/len(Evaluation['misclassified'])
            # Caclulate the error
            err = np.sum(Evaluation['weights']*Evaluation['misclassified'])/np.sum(Evaluation['weights'])
 
   
            # Calculate the alpha values
            alpha = np.log((1-err)/err)*0.5
            alphas.append(alpha)
            # Update the weights wi --> These updated weights are used in the sample_weight parameter
            # for the training of the next decision stump.
           
            '''
            j = 0
            for i in locations:
                self.dataset.weights.iloc[i] *= np.exp(alpha*Evaluation.misclassified.iloc[j])
                j += 1
            '''
            
            Y = self.dataset[self.target]
            data = self.dataset.drop([self.target], axis = 1)
            result = model.predict(data)
            
            for i in range(len(self.dataset)):
                if Y[i] != result[i]:
                    self.dataset['weights'].iloc[i] *= np.exp(alpha*result[i])
                
            
            # locatins #Evaluation['weights'] *= np.exp(alpha*Evaluation['misclassified'])
            #print('The Accuracy of the {0}. model is : '.format(t+1),accuracy*100,'%')
            #print('The missclassification rate is: ',misclassification*100,'%')
        self.dataset = self.dataset.drop(['weights','locations'], axis = 1)
        self.alphas = alphas
        self.models = models
            
    def predict(self):
        
        X_test = self.test_dataset.drop([self.target],axis=1).reindex(range(len(self.test_dataset)))
        Y_test = self.test_dataset[self.target].reindex(range(len(self.test_dataset))).where(self.dataset[self.target]==1,-1)
    
        # With each model in the self.model list, make a prediction 
        
        accuracy = []
        predictions = []
        
        for alpha,model in zip(self.alphas,self.models):
            prediction = alpha*model.predict(X_test) # We use the predict method for the single decisiontreeclassifier models in the list
            predictions.append(prediction)
            self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==Y_test.values)/len(predictions[0]))
           
        self.predictions = np.sign(np.sum(np.array(predictions),axis=0))
   
        
        
######Plot the accuracy of the model against the number of stump-models used##########

number_of_base_learners = 15
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

dataset = pd.read_excel('HSR.xlsx')
dataset = dataset.iloc[:,1:]
names = dataset.columns
dataset[names[0]] = np.where(dataset[names[0]] == 1, 1,-1)
target = names[0]

for i in range(number_of_base_learners):
    model = Boosting(dataset,i,dataset,target)
    model.fit()
    model.predict()
ax0.plot(range(len(model.accuracy)),model.accuracy,'-b')
ax0.set_xlabel('# models used for Boosting ')
ax0.set_ylabel('accuracy')
print('With a number of ',number_of_base_learners,'base models we receive an accuracy of ',model.accuracy[-1]*100,'%')    
                 
plt.show()        