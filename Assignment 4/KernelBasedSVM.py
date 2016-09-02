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
    
    # Another Dataset
    data = datasets.load_iris()
    x = data.data[:, :2]
    y = data.target
    idx = np.array(np.ix_(y == 2)).flatten()
    X = np.delete(x, idx, axis=0)
    Y = np.delete(y, idx, axis=0)
    for i in range(len(Y)):
        if (Y[i] == 0):
            Y[i] = -1
    
    # get X and Y
    X, Y = genDataset(100)
    
    # scaling data
    X = pp.scale(x)
    X = X1
    
    # shuffling the data
    sh = np.random.permutation(len(X))
    X = X[sh]
    Y = Y[sh]
    
    
    # Gaussian 
    # getting predicted and expected class
    predicted, expected = getCrossValidation(X, Y, 1e-8, None, 'Gauss')
    
    # Polynomial
    predicted, expected = getCrossValidation(X, Y, 1e-8, None, 'Poly')
    
    # counting number of labels classified correctly
    correct = np.sum(predicted == expected)
    
    
    # print results
    print ("Correct %d out of predicted %d: {}\n".format(correct, expected))
    
    # generate plots
    plt.figure(figsize=(12, 12))
    #plt.plot([1, 3])
    plt.subplot(311)
    plt.title('Data Distribution - Linearly-Separable Dataset')
    genPlot(X, Y)
    plt.subplot(312)
    plt.title('Gaussian - Hard Margins')
    plotLinear(X, Y, wh1, w0h1, svh1)
    plt.subplot(313)
    plt.title('Gaussian - Soft Margins')
    plotLinear(X, Y, ws1, w0s1, svs1)
    #plt.subplot(224)
    #plt.title('Result from Existing Function')
    #plotLinear(X, Y, w1, clf.intercept_, clf.support_vectors_)
    
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