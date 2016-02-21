"""
Created on Mon Feb 15 09:12:23 2016

@author: Parikshita
"""
# import libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as pp

# create function to load and plot data set
# the function take file name as input and returns x, y and opens the plot

def load_data(file_name):
    
    # Load Single Feature data set
    load_data = np.loadtxt(file_name, unpack=True)
    
    # Check the scale of the dataset if True store values of x and y 
    if(True):
        pp.scale(load_data)
        x = load_data[0]
        y = load_data[1]
    
    # reshaping x
    x = x.reshape(x.size,1) 
    
    return x, y
    
# plot to see the data distribution
def plot_data(x, y, filename):
    
    # Removing extension of the file
    strp_name = filename.strip('.dat')    
    
    # plot the file
    plt.figure()
    plt.scatter(x, y, color = 'black')
    plt.title("Data Distribution "+ strp_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('%s.png' % strp_name)
    
# plot for gradient descent
def plot_grad(Z, y, theta, filename):
    
    # Removing extension of the file
    strp_name = filename.strip('.dat')    
    
    for i in range(Z.shape[1]):
        y_hat = theta[0] + theta[1]*Z 
    
    plt.figure()
    plt.scatter(Z[:,1], y)
    plt.xlabel("Matrix Z")
    plt.ylabel("Original Y values")
    plt.title("Gradient Fitted Model of %s" % strp_name)
    plt.plot(Z, y_hat, 'k-')
    plt.savefig('gd%s.png' % strp_name)
    