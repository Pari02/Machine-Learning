"""
Created on Mon Feb 15 09:12:23 2016

@author: Parikshita
"""

# import libraries
import numpy as np
from numpy import linalg as la
from sklearn import preprocessing as pp
from sklearn.cross_validation import KFold
import random

# function to calculate training and test datasets
def train_test(Z, y):
    x_train = Z[:-20]
    x_test = Z[-20:]
    y_train = y[:-20]
    y_test =  y[-20:]
    return x_train, x_test, y_train, y_test

# create matrix Z by appending column of 1's to x
def matrixZ(x, degree):
    poly = pp.PolynomialFeatures(degree)
    Z = poly.fit_transform(x)
    return Z

# function to calculate parameter(s), theta
def find_theta(Z, y):
    theta = np.dot(la.pinv(Z), y)
    return theta

# function to calculate mean square error
def mse_err(Z, y, theta):
    err = np.dot(Z, theta) - y
    J_theta = np.dot(np.transpose(err), err)
    mean_err = J_theta/len(Z)    
    return mean_err

# function to peform 10 Fold cross validation 
def kFoldsValidation(x, y, folds, m):
    kf = KFold(m, folds)
    tot_error = list()
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        theta = find_theta(x_train, y_train)
        tot_error.append(mse_err(x_train, y_train, theta))
    return np.mean(tot_error)
    
# function to implement gradient descent algorithm    
def gradient_descent(Z, y, eta, numIterations, m, n):
    theta = []
    itr = 0
    for i in xrange(n):
        theta.append(random.random())
    while itr < numIterations:
        h_theta = np.dot(Z, theta)
        err = h_theta - y
            
        # calculating gradient cost 
        # J_theta = np.sum(err ** 2) / (2 * m) 
         
        gradient = np.dot(Z.transpose(), err) / m         
        theta = theta - eta * gradient
        itr = itr + 1
    return theta