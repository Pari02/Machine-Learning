# Author: Parikshita Tripathi
# Time : 03/27/2016

#--------------------------------------------------------------------------#

# import packages
from __future__ import division
from math import *
import numpy as np
from numpy import linalg as la
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing as pp
import logistic_regression
reload(logistic_regression)
from logistic_regression import *


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
def getResults(train_x, train_y, test_x, test_y, xClass, classes):
    
    TP, TN, FP, FN = getConfMatrixParam(classes, test_y, xClass)
    
    # print confusion matrix
    conf_matrix = confusion_matrix(test_y, classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # print performance parameters
    accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
    fMeasure = getFMeasure(precision, recall)
    
    print ("Accuracy: {}\n ".format(accuracy))
    print ("Precision: {}\n".format(precision))
    print ("Recall: {}\n".format(recall))
    print ("F-Measure: {}".format(fMeasure))    
    
            
    
        