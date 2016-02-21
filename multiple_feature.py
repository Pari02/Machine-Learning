"""
Created on Mon Feb 15 09:12:23 2016

@author: Parikshita
"""

# import libraries
import numpy as np
from sklearn import linear_model as lm
import load_plot, regr
reload(load_plot)
reload(regr)
from load_plot import *
from regr import *
from os import listdir

def main():

    # store names of single and multiple variable files in a variables
    m_filename = listdir('MFF')
    
    file_name = []
    for f in m_filename:
        if f.endswith('.dat'):
            file_name.append(f)

    for i, s in enumerate(file_name):
        # call function to read file and generate the plot
        x, y = load_data("".join(('MFF/', s)))
        
        # Calling function to create Z matrix 
        Z = matrixZ(x, 1)

        # getting rowns and columns of x
        m, n = np.shape(Z) 
    
        # fit the linear model
        theta = find_theta(Z, y)

        # calculate regression using pre-defined python function
        linear_regr = lm.LinearRegression()

        # training model using the training sets
        linear_regr.fit(Z, y)
        
        # Mean square error calculation by 10 Fold cross validation 
        kf_err = kFoldsValidation(Z, y, 10, m)
        
        # assigning value to learning rate = eta and number of iteration = itr
        eta = 0.001
        itr = 100000
    
        # calling gradient descent function to get parameter values
        g_theta = gradient_descent(Z, y, eta, itr, m, n)
        
        # mean square error after performing gradient descent
        gd_mse = mse_err(Z, y, g_theta)
        
        # Printing Results: -
        # parameterds
        print('Printing Output for Multiple Feature File dataset {}: -'.format(s))
        print('\n')
        print("Parameters by Computed Model: {}\n".format(theta))
        print("Parameters by Pre-defined function:\n Coefficients: {}\n Intercept: {}\n".format(linear_regr.coef_, linear_regr.intercept_))
    
        
        # Error calculation via 10 Fold cross validation
        print("Mean Square error by 10 Fold cross validation: {}\n".format(kf_err))   
        
        # Evalutation results from gradient descent
        print("Parameters by Gradient Descent: {}".format(g_theta))
        print("Mean Square Error by Gradient Descent: {}".format(gd_mse))
        print('*****************************************************************')
        
        
if __name__ == "__main__":
    main()
