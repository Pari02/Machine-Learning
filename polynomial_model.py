# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:12:23 2016

@author: Parikshita

"""

# import libraries
import numpy as np
from sklearn import linear_model as lm
from load_plot import *
from regr import *

# create polynomial_model 
def main():
    
    s_filename = ['svar-set1.dat', 'svar-set2.dat', 'svar-set3.dat', 'svar-set4.dat']
    
    for i, s in enumerate(s_filename):
        # call function to read file and generate the plot
        x, y = load_data("".join(('SFF/', s)))
        
        m = y.size
        # removing extension from the file name
        strp_name = s.strip('.dat')
        
        fig, axs = plt.subplots(3, 1, figsize=(15, 6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.001)

        for i, degree in enumerate(xrange(2,5)):
            Z = matrixZ(x, degree)
            theta = find_theta(Z, y)
            kf_err = kFoldsValidation(Z, y, 10, m)
            axs[i].scatter(x, np.dot(Z, theta), cmap = plt.cm.Oranges)
            axs[i].set_title(strp_name + ', degree: ' + str(degree))
            
            print("MSE of file {} with degree {} by 10 Fold cross validation: {}".format(strp_name, degree, kf_err))
        fig.savefig('%s.png' % "".join(('poly',strp_name)))

if __name__ == "__main__":
    main()
    