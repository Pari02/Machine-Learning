# Author: Parikshita Tripathi
# Time : 03/08/2016

#--------------------------------------------------------------------------#

# import packages
import loadData
import Computation
reload(loadData)
reload(Computation)
from loadData import *
from Computation import *
import pylab as pl
import matplotlib.pyplot as plt


def main():
    filename = '/Users/deepakkuletha/Desktop/Data Science/Spring 2016/Machine Learning/Homework/Assignment 2/pima-indians-diabetes.txt'
    
    x, y = load_data(filename)
    x1 = x[:, np.newaxis,0] # extracting one feature
    
    train_x, test_x, train_y, test_y = getCrossValidation(x1, y)
    
    classes = np.unique(y)
    # print model paramenters
    for j in range(len(classes)): 
        mu, sigma, alpha = getMuSigmaAlpha(train_x, train_y, j)
        print("Mean of class {}: {}\n".format(j,mu))
        print("Sigma of class {}: {}\n".format(j, sigma))
        print("Prior class probability of class {}: {}\n".format(j, alpha))
    
    # print output of discriminant function
    print getDiscriminantFuncTwoClass(train_x, train_y, test_x)

    # get all predicted class labels
    predictClass = getEachClassClassification(train_x, train_y, test_x)
    
    # get the final classification of the point
    # and use it to get elements of confusion matrix
    xClass = getXClassification(train_x, train_y, test_x, 1.0)
    TP, TN, FP, FN = getConfMatrixParam(predictClass, test_y, xClass)
    
    # print confusion matrix
    conf_matrix = confusion_matrix(test_y, predictClass, labels = classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # plotting confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    pl.title('Confusion matrix of 1D 2-Class Classifier')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()
	    
    # print performance parameters
    accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
    fMeasure = getFMeasure(precision, recall)
    
    print ("Accuracy: {}".format(accuracy))
    print ("Precision: {}".format(precision))
    print ("Recall: {}".format(recall))
    print ("F-Measure: {}".format(fMeasure))    
    
    
if __name__ == "__main__":
    main()
    