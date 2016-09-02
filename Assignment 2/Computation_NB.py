# Author: Parikshita Tripathi
# Time : 03/08/2016

#--------------------------------------------------------------------------#

# import packages
from __future__ import division
from math import *
from sklearn.naive_bayes import MultinomialNB
import numpy as np

#---------------------------------------------------------------------
# function to find model parameters
def getModelParam(X, Y, j):
    
    # calculate total examples in class j 
    # and get all examples in each class
    mJ = X[np.ix_(Y == j)]
    sum_mJ = len(mJ)
    
    # probabilty of prior 
    alphaJ = sum_mJ/(len(Y))

    # calcualte mu
    muJ = np.mean(mJ, axis = 0)
    
    return muJ, alphaJ 
            
            
#---------------------------------------------------------------------
# function to calculate membership function for Naive Bayes - Bernoulli case
def getMemFunc(train_x, train_y, test_x, j):
    
    mu, alpha = getModelParam(train_x, train_y, j)
    
    gJx = 0
    for i in range(len(mu)):
        gJx += (test_x[i] * math.log(mu[i])) + ((1 - test_x[i]) * (1 - mu[i])) + math.log(alpha)
    return gJx
         
               
#---------------------------------------------------------------------  
# function to compute label of each class          
def getEachClassClassification(X, Y):
    
    classes = np.unique(Y)
    num_rows = np.shape(X)[0]
    kClass = []
    
    for i in range(num_rows):
        maxClass = None
        maxMemFunc = None
        for j in classes: 
            gJx = getMemFunc(X, Y, X[i,:], j)
            #print ("Membership Function class {}: {}".format(j, gJx))
            if maxMemFunc is None:
                maxMemFunc = gJx
                maxClass = j
            elif gJx > maxMemFunc:
                maxMemFunc = gJx
                maxClass = j
        kClass.append(maxClass)

    return kClass


#---------------------------------------------------------------------   
# function to generate model for Naive Bayes Binomial
def getNBBinomial(X, Y):
     # get classifier for Naiye Bayes Binomial 
     clf = MultinomialNB()
     # generate model
     gen_model = clf.fit(X, Y)
     # predict labels for each class
     predictClass = gen_model.predict(X)
     return predictClass
            