# Author: Parikshita Tripathi
# Time : 04/07/2016

#--------------------------------------------------------------------------#
# import packages
import neuralNetworks
reload(neuralNetworks)
from SVM import *
import generatePlots
reload(generatePlots)
from generatePlots import *
from sklearn import preprocessing as pp
import numpy as np

def main():
    
   # get X and Y
    X, Y = genDataset(100)
    
    # scaling data
    X = pp.scale(x)
    X = X1
    
    # shuffling the data
    sh = np.random.permutation(len(X))
    X = X[sh]
    Y = Y[sh]
    
    # getting predicted and expected class using Linear SVM with Hard Margins
    predicted, expected = getCrossValidation(X, Y, 1e-8, None, 'Linear')
    
    # getting predicted and expected class using Linear SVM with Soft Margins
    # predicted, expected = getCrossValidation(X, Y, 1e-8, 0.1, 'Linear')
    
    # counting number of labels classified correctly
    correct = np.sum(predicted == expected)
    
    
    # print results
    print ("Correct %d out of predicted %d: {}\n".format(correct, expected))
      
    # generate plots
    plt.figure(figsize=(12, 12))
    #plt.plot([1, 3])
    plt.subplot(221)
    plt.title('Data Distribution - Linearly-Separable Dataset')
    genPlot(X, Y)
    plt.subplot(222)
    plt.title('Result from Implemented Algorithm - Hard Margins')
    plotLinear(X, Y, wh, w0h, svh)
    plt.subplot(223)
    plt.title('Result from Implemented Algorithm - Soft Margins')
    plotLinear(X, Y, ws, w0s, svs)
    plt.subplot(224)
    plt.title('Result from Existing Function')
    plotLinear(X, Y, w1, clf1.intercept_, clf1.support_vectors_)
    
# output
#Error of predicted output: -2.0
#
#Weights:
#Weight from calculation: [-60.30108094  53.506688  ]
#Weight from existing Function: [[-1.81047531  1.46854549]]
#
#Intercept:
#Intercept from calculation: 0.742028513931
#Intercept from existing Function: [ 0.20457366]

    
if __name__ == "__main__":
    main()