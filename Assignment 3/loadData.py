# Author: Parikshita Tripathi
# Time : 03/27/2016

#--------------------------------------------------------------------------#

# import packages
import numpy as np

# function loads the file and separates features and class variable
def load_data(filename):
    
    # Load Single Feature data set
    load_data = np.genfromtxt(filename, delimiter= ',')
    
    # getting column and row information of dataset
    c = load_data.shape[1]
    
    x = load_data[:,xrange(0,c-1)] # features
    y = load_data[:,c-1]  # class variable
    
    return x, y
    
    
# filename = '/Users/deepakkuletha/Desktop/Data Science/Spring 2016/Machine Learning/Homework/Assignment 2/pima-indians-diabetes.txt'
# filename = '/Users/deepakkuletha/Desktop/Data Science/Spring 2016/Machine Learning/Homework/Assignment 2/spambase.txt'
    