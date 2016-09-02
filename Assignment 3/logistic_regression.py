# Author: Parikshita Tripathi
# Time : 03/27/2016

#--------------------------------------------------------------------------#

# import packages
from __future__ import division
import numpy as np
from numpy import linalg as la
from math import log
from sklearn import preprocessing as pp
from sklearn.cross_validation import KFold
import random


#---------------------------------------------------------------------
# function to append 1s in the first column of x
def appendOne(x):
    return np.append(np.ones([np.shape(x)[0], 1]), x, 1)
    
    
#---------------------------------------------------------------------    
# function to map feature
def mapFeatures(x, d):
    if d == 1:
        Z = x
    else:
        poly = pp.PolynomialFeatures(d)
        Z = poly.fit_transform(x)
    return Z

#---------------------------------------------------------------------
# function to generate values of initial theta
def initTheta(n):
    #theta = np.zeros([n])
    theta = []
    for i in range(n):
        theta.append(random.uniform(0, 0.005))
        
    return np.reshape(theta, (n))

    

#---------------------------------------------------------------------
# function to get boolean value for indicator function
def indFunct(y, j):
    if (y == j):
        return 1
    else:
        return 0
    
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
# function to calculate sigmoid function
# where we pass theta and x = x(i) as parameters
# and x(i) is the ith x vector
def sigmoidFunc(theta, x):
    
    # calculate dot product of theta transpose and x
    thetax = np.dot(np.transpose(theta), x)
    # calculate sigmoid function
    h_thetax = 1.0 / (1 + np.exp(-thetax))
    
    return h_thetax


#---------------------------------------------------------------------
# function to calculate softmax function
# here we calculate hypothesis for 'j' class
def softmaxFunc(theta, x, y, j):
    
    thetaxJ = np.dot(np.transpose(theta), x)
    h_thetax = thetaxJ / thetaxI
    
    return h_thetax

def softmaxDen(theta, x):
    
    sum = 0
    for i in range(np.shape(x)[0]):
        sum += np.exp(np.dot(np.transpose(theta[i]), x[i]))
    return sum
    
   
#---------------------------------------------------------------------       
# this function calculates the log likelihood
def calCost(theta, train_x, train_y):
 
    sigmoidx = sigmoidFunc(theta, train_x)
    if sigmoidx == 0:
        sum = 0
    else:
        sum = (train_y * log(sigmoidx)) + ((1 - train_y) * log(1 - sigmoidx))
        
    return sum
          
                 
#---------------------------------------------------------------------
def calGradientDescent(train_x, train_y, eta, numIterations):
    
    # inititalize variables
    sum = 0
    itr = 0
    # get values of number of examples, m and number of features, n
    m, n = np.shape(train_x)
    
    # get initial value of theta 
    theta = initTheta(n)
    
    while itr < numIterations:
        
        for i in range(m):
            
            # get sigmoid function of ith x value
            sigmoidx = sigmoidFunc(theta, train_x[i])
            sum += (sigmoidx - train_y[i]) * train_x[i]   
            
        newTheta = theta - (eta * sum)
        itr = itr + 1
        theta = newTheta
        
    return theta

#---------------------------------------------------------------------
def calGradientDescentKClass(train_x, train_y, eta, numIterations, j):
    
    # inititalize variables
    sum = 0
    itr = 0
    # get values of number of examples, m and number of features, n
    m, n = np.shape(train_x)
    
    # get initial value of theta 
    theta = initTheta(n)
    
    while itr < numIterations:
        
        for i in range(m):
            
            # get sigmoid function of ith x value
            softmaxj = softmaxFunc(theta, train_x[i], train_y[i], j)
            sum += (softmaxj - indFunct(train_y[i], j)) * train_x[i]   
            
        newTheta = theta - (eta * sum)
        itr = itr + 1
        theta = newTheta
        
    return theta
    
    
#---------------------------------------------------------------------
# return boolean value if h_theta is greater than 1 for a class j
def xClass(h_theta):
    return h_theta > 0.5
    
                
#---------------------------------------------------------------------
# predict labels with k class dataset
def getClassClassification(theta, test_x, test_y):
    
    m = np.shape(test_x)[0]
    
    kClass = []
    for i in range(m):
        hTheta = sigmoidFunc(theta, test_x[i])
        kClass.append(xClass(hTheta))

    return kClass
 