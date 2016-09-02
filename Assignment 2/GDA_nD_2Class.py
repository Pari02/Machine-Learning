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
    
    train_x, test_x, train_y, test_y = getCrossValidation(x2, y)
    
    classes = np.unique(y)

    # get all predicted class labels
    predictClass = getEachClassClassification(train_x, train_y, test_x)
            
    # print model paramenters
    for j in range(len(classes)): 
        mu, sigma, alpha = getMuSigmaAlpha(train_x, train_y, j)
        
        # get the final classification of the point
        # and use it to get elements of confusion matrix
        TP, TN, FP, FN = getConfMatrixParam(predictClass, test_y, j)
        
        # print performance parameters
        accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
        fMeasure = getFMeasure(precision, recall)
        
        print("Mean of class {}: {}\n".format(j,mu))
        print("Sigma Matrix of class {}:\n {}\n".format(j, sigma))
        print("Prior class probability of class {}: {}\n".format(j, alpha))   
        print ("Accuracy for class {}: {}".format(j, accuracy))
        print ("Precision for class {}: {}".format(j, precision))
        print ("Recall for class {}: {}".format(j, recall))
 	print ("F-Measure for class {}: {}".format(j, fMeasure))
   
    # print output of discriminant function
    print getDiscriminantFuncTwoClass(train_x, train_y, test_x)

    # get all predicted class labels
    predictClass = getEachClassClassification(train_x, train_y, test_x)
       
    # print confusion matrix
    conf_matrix = confusion_matrix(test_y, predictClass, labels = classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # plotting confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    pl.title('Confusion matrix of nD 2-Class Classifier')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()
	    
   # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y, getEachClassClassification(x, y, x))
        average_precision[i] = average_precision_score(y, getEachClassClassification(x, y, x))

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], label = 'Precision-recall curve (area = {0:0.2f})'''.format(average_precision[0]))
    
    for i in range(len(classes)):
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve nD 2 - Class'.format(average_precision[0]))
        plt.legend(loc="lower left")
        plt.show()

    
if __name__ == "__main__":
    main()
    