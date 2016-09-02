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
from sklearn.metrics import precision_recall_curve, average_precision_score


def main():
    filename = '/Users/deepakkuletha/Desktop/Data Science/Spring 2016/Machine Learning/Homework/Assignment 2/pima-indians-diabetes.txt'
    
    x, y = load_data(filename)
    x2 = x[:, (0,1)] # extracting one feature
    
    train_x, test_x, train_y, test_y = getCrossValidation(x2, y)
    
    classes = np.unique(y)
    # print model paramenters
    for j in range(len(classes)): 
        mu, sigma, alpha = getMuSigmaAlpha(train_x, train_y, j)
        print("Mean of class {}: {}\n".format(j,mu))
        print("Sigma Matrixof class {}:\n {}\n".format(j, sigma))
        print("Prior class probability of class {}: {}\n".format(j, alpha))
    
    # print output of discriminant function
    print getDiscriminantFuncTwoClass(train_x, train_y, test_x)

    # get all predicted class labels
    predictClass = getEachClassClassification(train_x, train_y, test_x)
    
    # get the classification of the point using class j
    # and use it to get elements of confusion matrix
    TP, TN, FP, FN = getConfMatrixParam(predictClass, test_y, 1)
    
    # print confusion matrix
    conf_matrix = confusion_matrix(test_y, predictClass, labels = classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # plotting confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    pl.title('Confusion matrix of 2D 2-Class Classifier')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()
	    
    # print performance parameters
    accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
    fMeasure = getFMeasure(precision, recall)
    
    print ("Accuracy: {} ".format(accuracy))
    print ("Precision: {}".format(precision))
    print ("Recall: {}".format(recall))
    print ("F-Measure: {}".format(fMeasure))   
    
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y, getEachClassClassification(x2, y, x2))
        average_precision[i] = average_precision_score(y, getEachClassClassification(x2, y, x2))

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], label = 'Precision-recall curve (area = {0:0.2f})'''.format(average_precision[0]))
    
    for i in range(len(classes)):
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve 2D 2 - Class'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()

    
if __name__ == "__main__":
    main()
    