# -*- coding: utf-8 -*-
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
    s_filename = listdir('SFF')
    
    file_name = []
    for f in s_filename:
        if f.endswith('.dat'):
            file_name.append(f)

    for i, s in enumerate(file_name):
        # call function to read file and generate the plot
        x, y = load_data("".join(('SFF/', s)))

        # Calling function to create Z matrix 
        Z = matrixZ(x, 1)

        # getting rowns and columns of x
        m, n = np.shape(Z) 

        # Split data into training and testing datasets
        x_train, x_test, y_train, y_test = train_test(Z, y)
    
        # fit the linear model
        theta = find_theta(x_train, y_train)

        # find training and testing errors i.e Mean square errors
        # training error
        err_train = mse_err(x_train, y_train, theta)
    
        # testing error
        err_test = mse_err(x_test, y_test, theta)

        # calculate regression using pre-defined python function
        linear_regr = lm.LinearRegression()

        # training model using the training sets
        linear_regr.fit(x_train, y_train)

        # calculate mse via pre-defined functions (pdf)
        pdf_mse_train = np.mean((linear_regr.predict(x_train) - y_train) ** 2)
        pdf_mse_test = np.mean((linear_regr.predict(x_test) - y_test) ** 2)
        
        
        # Mean square error calculation by 10 Fold cross validation 
        kf_err = kFoldsValidation(Z, y, 10, m)
        
        # assigning value to learning rate = eta and number of iteration = itr
        eta = 0.001
        itr = 100000
    
        # calling gradient descent function to get parameter values
        g_theta = gradient_descent(Z, y, eta, itr, m, n)
        
        # mean square error after performing gradient descent
        gd_mse = mse_err(x_test, y_test, g_theta)
        # plotting 
        
        # Printing Results: -
        # parameterds
        print('Printing Output for Single Feature File dataset {}: -'.format(s))
        print('\n')
        print("Parameters by Computed Model: {}\n".format(theta))
        print("Parameters by Pre-defined function:\n Coefficients: {}\n Intercept: {}\n".format(linear_regr.coef_, linear_regr.intercept_))

        # training error
        print("Training Error by Computed Model: {}\n".format(err_train))
        print("Training Error by Pre-defined function: {}\n".format(pdf_mse_train))

        # testing error
        print("Testing Error by Computed Model: {}\n".format(err_test))
        print("Training Error by Pre-defined function: {}\n".format(pdf_mse_test))
    
        
        # Error calculation via 10 Fold cross validation
        print("Mean Square error by 10 Fold cross validation: {}\n".format(kf_err))   
        
        # Evalutation results from gradient descent
        print("Parameters by Gradient Descent: {}".format(g_theta))
        print("Mean Square Error by Gradient Descent: {}".format(gd_mse))
        print('*****************************************************************')

        # plot data
        plot_data(x, y, s)
        
        # plotting result of gradient descent
        plot_grad(Z, y, g_theta, s)
        

if __name__ == "__main__":
    main()