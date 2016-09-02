# Author: Parikshita Tripathi
# Time : 03/08/2016

#--------------------------------------------------------------------------#

# import packages
import Computation
reload(Computation)
from Computation import *
import pylab as pl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def main():
    iris = datasets.load_iris()
    
    x = iris.data
    y = iris.target 
    
    train_x, test_x, train_y, test_y = getCrossValidation(x, y)
    
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

    
   # print confusion matrix
    conf_matrix = confusion_matrix(test_y, predictClass, labels = classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # plotting confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    pl.title('Confusion matrix of nD k-Class Classifier')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()

   # accuracy = accuracy_score(y,getEachClassClassification(x, y, x))
   # precision = precision_score(y,getEachClassClassification(x, y, x))
    
    
if __name__ == "__main__":
    main()
    