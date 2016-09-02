# Author: Parikshita Tripathi
# Time : 03/27/2016

#--------------------------------------------------------------------------#

# import packages
from sklearn import datasets 
import pylab as pl 
import logistic_regression
import Computation
reload(logistic_regression)
reload(Computation)
from logistic_regression import *
from Computation import *


def main():
    # load data
    dataSet = datasets.load_digits(n_class = 2)
    #dataSet = datasets.load_iris()

    x = dataSet.data
    y = dataSet.target
    
    newX = appendOne(x)
    
    # implement algorithm for non-linear inputs
    Z = mapFeatures(newX, 3)
    train_x, test_x, train_y, test_y = getCrossValidation(Z, y)
    
    # assigning value to learning rate = eta and number of iteration = itr
    eta = 0.001
    itr = 100
    
    theta = calGradientDescentKClass(train_x, train_y, eta, numIterations, 1)
    print("Parameter theta values are \n{}".format(theta))

    classes = getClassClassification(theta, test_x, test_y)
    print("Classification: \n {}".format(classes))
    print("Original y: \n{}".format(test_y))    
    error = classes - test_y
    print("Error: \n{}".format(error))

    
    getResults(train_x, train_y, test_x, test_y, 0, classes)
    
    

    # plot the digits
    pl.gray() 
    pl.matshow(digits.images[0]) 
    pl.show() 
    
if __name__ == "__main__":
    main()