# Author: Parikshita Tripathi
# Time : 03/08/2016

#--------------------------------------------------------------------------#

# import packages
from __future__ import division
from math import *
import numpy as np
from numpy import linalg as la
from sklearn import preprocessing as pp
from sklearn.cross_validation import KFold


#---------------------------------------------------------------------
# function to do cross validation
def getCrossValidation(X, Y):
    kf = KFold(len(X), 10)
    
    for trainID, testID in kf:
        train_x = X[trainID]
        test_x = X[testID]
        train_y = Y[trainID]
        test_y = Y[testID]
    
    return train_x, test_x, train_y, test_y


#---------------------------------------------------------------------
# function to compute mu, sigma and alpha( probability of prior)
# input variables are X = train_x, Y = train_y, j = class
def getMuSigmaAlpha(X, Y, j):
    
    # calculate total examples in class j 
    # and get all examples in each class
    mJ = X[np.ix_(Y == j)]
    
    sum_mJ = len(mJ)
        
    # calcualte mu, sigma and alpha
    muJ = np.mean(mJ, axis = 0)
    sigmaJ = np.cov(mJ.T)
    alphaJ = sum_mJ/(len(Y))
    
    return muJ, sigmaJ, alphaJ 


#---------------------------------------------------------------------
## function for membership function for 1D 
def getMemFunc(train_x, train_y, test_x, j):
    
    mu, sigma, alpha = getMuSigmaAlpha(train_x, train_y, j)
    
    # gJx is membership function
    if(sigma.size == 1): # for 1D
        exp_part = - np.dot((test_x - mu).T, (test_x - mu))/ (2 * sigma) 
        gJx = (- log(sqrt(sigma))) + exp_part + log(alpha)
    else: # for nD
        exp_part = - np.dot(np.dot((test_x - mu).T, 1/sigma), (test_x - mu))/ 2
        gJx = (- log(la.det(sigma))) + exp_part + log(alpha)
            
    return gJx


#---------------------------------------------------------------------
# classifying classes by applying discriminant function
def getDiscriminantFuncTwoClass(train_x, train_y, test_x):  
    
    num_rows = np.shape(test_x)[0]
    for i in range(num_rows):
        dx = getMemFunc(train_x, train_y, test_x[i,:], 1.0) - getMemFunc(train_x, train_y, test_x[i,:], 0.0)    
        if (dx > 0):
            result =  str("Output from discriminat function, Y-hat: {}\n".format(1))
        else:
            result =  str("Output from discriminat function, Y-hat: {}\n".format(0))
        return result


#---------------------------------------------------------------------
# function to predict labels with 2 class dataset
def getXClassification(train_x, train_y, test_x, j):
    
    classes = np.unique(train_y)
    
    maxClass = None
    maxMemFunc = None
    for j in classes: 
        gJx = getMemFunc(train_x, train_y, test_x, j)
        
        if maxMemFunc is None:
            maxMemFunc = gJx
            maxClass = j
        elif gJx > maxMemFunc:
            maxMemFunc = gJx
            maxClass = j
        
    return maxClass

                
#---------------------------------------------------------------------
# predict labels with k class dataset
def getEachClassClassification(train_x, train_y, test_x):
    
    classes = np.unique(train_y)
    num_rows = np.shape(test_x)[0]
    
    kClass = []
    for i in range(num_rows):
        maxClass = None
        maxMemFunc = None
        for j in classes: 
            gJx = getMemFunc(train_x, train_y, test_x[i,:], j)
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
# calculate true positive TP, true negative TN, fake positive FP
# and fake negative FN
def getConfMatrixParam(predictClass, test_y, j):
    
    num_rows = np.shape(test_y)[0]
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(num_rows):       
        if (predictClass[i] == j and test_y[i] == j):
            TP += 1 
        if (predictClass[i] != j and test_y[i] != j):
            TN += 1      
        if (predictClass[i] == j and test_y[i] != j):
            FP += 1  
        if (predictClass[i] != j and test_y[i] == j):
            FN += 1 
   
    return TP, TN, FP, FN
        
#---------------------------------------------------------------------        
def getPerfomanceParam(TP, TN, FP, FN):
    
   # calculate accuracy, precision and recall
    accuracy = (TP + TN)/ (TP + TN + FP + FN)
    
    # calculate precison and recall
    if (TP == 0): 
        precision = 0
    else:
        precision = TP/(TP + FP)
    
    if (TP == 0): 
        recall = 0
    else:
        recall = TP/(TP + FN)
    
    return accuracy, precision, recall


#---------------------------------------------------------------------
# function to calculate F-measure
def getFMeasure(precision, recall):
    if (precision == 0 and recall == 0):
        fMeasure = 0
    else:
        fMeasure = (2 * (precision + recall))/ (precision + recall)
    return fMeasure

#---------------------------------------------------------------------
# function to print final results
def getResults(X, Y):
    
    # x1 = x[:, np.newaxis,0]
    #x2 = x[:, np.newaxis, (0,1)]
    
    # call KFold
    train_x, test_x, train_y, test_y = getCrossValidation(X, Y)
    
    # print model paramenters
    mu, sigma, alpha = getMuSigmaAlpha(train_x, train_y, j)
    print("Mean of each class: {}\n".format(mu))
    print("Sigma matrix: {}\n".format(sigma))
    print("Prior class probability: {}\n".format(mu))
    
    # print output of discriminant function
    print getDiscriminantFunc(train_x, train_y, test_x)
    
    # get all predicted class labels
    predictClass = getEachClassClassification(train_x, train_y, test_x)
    
    # get the final classification of the point
    # and use it to get elements of confusion matrix
    xClass = getXClassification(train_x, train_y, test_x)
    TP, TN, FP, FN = getConfMatrixParam(predictClass, test_y, xClass)
    
    # print confusion matrix
    conf_matrix = confusion_matrix(test_y, predictClass)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # print performance parameters
    accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
    fMeasure = getFMeasure(precision, recall)
    
    print ("Accuracy: {}\n ".format(accuracy))
    print ("Precision: {}\n".format(precision))
    print ("Recall: {}\n".format(recall))
    print ("F-Measure: {}".format(fMeasure))    
    
            
    
        