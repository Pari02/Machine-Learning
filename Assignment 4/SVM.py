# Author: Parikshita Tripathi
# Time : 04/07/2016

#--------------------------------------------------------------------------#

# import packages
from __future__ import division
from math import *
import numpy as np
from numpy import linalg as la, random as rn
from sklearn import preprocessing as pp
from sklearn.cross_validation import KFold
from cvxopt import solvers, matrix
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import datasets, metrics as mt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

#---------------------------------------------------------------------
# functions to generate Gram matrix based on parameters
def getKernel(X1, X2, kernel):
    if kernel == 'Linear':
        #def getLinearKernel(X1, X2):
        GramM = np.dot(X1, X2)
        
    elif kernel == 'Gauss':
        #def getGaussKernel(X1, X2, sigma = 5.0):
        sigma = 5.0
        GramM = np.exp(-la.norm(X1 - X2) ** 2 / (2 * (sigma ** 2)))
        
    elif kernel == 'Poly':
        #def getPolyKernel(X1, X2, degree = 3):
        q = 3
        GramM = (np.dot(X1, X2) + 1) ** q
    return GramM

#---------------------------------------------------------------------
# generate 2D feature 2 class linearly separable dataset
# or non separable dataset
def genDataset(N, separable = True):
    n = N//2
    if separable:
        mu1 = np.array([0, 2])
        mu2 = np.array([2, 0])
        sigma = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = rn.multivariate_normal(mu1, sigma, N)
        Y1 = np.ones(len(X1))
        X2 = rn.multivariate_normal(mu2, sigma, N)
        Y2 = np.ones(len(X2)) * -1
        X = np.vstack((X1, X2))
        Y = np.hstack((Y1, Y2))
    else:
        X = np.random.randn(300, 2)
        Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

        mu1 = [-1, 2]
        mu2 = [1, -1]
        mu3 = [4, -4]
        mu4 = [-4, 4]
        sigma = [[1.0,0.8], [0.8, 1.0]]
        X1 = rn.multivariate_normal(mu1, sigma, n)
        X1 = np.vstack((X1, np.random.multivariate_normal(mu3, sigma, n)))
        Y1 = np.ones(len(X1))
        X2 = rn.multivariate_normal(mu2, sigma, n)
        X2 = np.vstack((X2, np.random.multivariate_normal(mu4, sigma, n)))
        Y2 = np.ones(len(X2)) * -1
        X = np.vstack((X1, X2))
        Y = np.hstack((Y1, Y2))
    return X, Y
       
#---------------------------------------------------------------------
def getCrossValidation(X, Y, E, c = None, kernel = 'Linear'):
    kf = KFold(len(X), 10)
    
    predicted = []
    for trainID, testID in kf:
        w, w0, sv_idx, sv = getSVM(X[trainID], Y[trainID], E, c, kernel)
        predicted.append(classifyLables(X[testID], w, w0))
    
    return predicted, Y   
           
#---------------------------------------------------------------------
# functions taakes parameters X, Y 
# and epsilon E (it is used to get support vectors)
def getSVM(X, Y, E, c, kernel):
    
    # get number of examples, m and number of feature vectors n 
    m,n = np.shape(X)
    
    # calculate Gram matrix
    GramM = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            GramM[i,j] = getKernel(X[i], X[j], kernel)

    # get parameters P, q, G, h, A, b to calculate alpha
    P = matrix(np.dot(Y, Y.T) * GramM)
    q = matrix(np.ones(m) * -1)
    #G = matrix(np.vstack((np.diag(np.ones(m) * -1), matrix(np.identity(m)))))
    if (c is None):
        G = matrix(np.diag(np.ones(m) * -1))
        h = matrix(np.vstack((matrix(np.zeros(m)), matrix(np.ones(m)))))
    else:
        G = matrix(np.vstack((np.diag(np.ones(m) * -1), matrix(np.identity(m)))))
        h = matrix(np.vstack((matrix(np.zeros(m)), matrix(np.ones(m) * c))))
    A = matrix(Y, (1, m), tc = 'd')
    b = matrix(0.0)
    
    # solve to get the alpha  
    sol = solvers.qp(P, q, G, h, A, b)
    
    alpha = np.ravel(sol['x'])
    
    #sv = []
    sv = alpha >  E
    
    # get indexes of support vectors
    sv_idx = np.arange(len(alpha))[sv]

    # get all alphas, X and Y values related to support vector
    # use them to calculate weight and intercept
    alpha_sv = alpha[sv]
    X_sv = X[sv]
    Y_sv = Y[sv]
    
    # now find weight, w and intercept, w0
    # initialize w and w0 to zero

    w = np.zeros(n)
    for i in range(len(alpha_sv)):
        w += alpha_sv[i] * Y_sv[i] * X_sv[i]
    
    w0 = 0
    for i in range(len(sv_idx)):
        w0 += (Y_sv[i] - np.dot(w, X_sv[i]))
        w0 = w0/len(sv_idx)
    
    return w, w0, sv_idx, X_sv

#---------------------------------------------------------------------
# function to classify
def classifyLabels(X, w, w0):
    labels = []
    
    for i in range(len(X)):
        cond = np.dot(w, X[i]) - w0
        if cond > 0:
            prLabel = 1
        else:
            prLabel = -1
        labels.append(prLabel)
    return labels
    
