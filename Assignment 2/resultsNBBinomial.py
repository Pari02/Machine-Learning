
# Author: Parikshita Tripathi
# Time : 03/08/2016

#--------------------------------------------------------------------------#

# import packages
import loadData
import Computation
import Computation_NB
reload(loadData)
reload(Computation)
reload(Computation_NB)
from loadData import *
from Computation import *
from Computation_NB import *


#---------------------------------------------------------------------
def main():
    filename = '/Users/deepakkuletha/Desktop/Data Science/Spring 2016/Machine Learning/Homework/Assignment 2/spambase.txt'
    
    x, y = load_data(filename)
    
    # get all predicted class labels
    predictClass = getNBBinomial(x, y)
    
    classes = np.unique(y)
     
    # print model parameters
    for j in range(len(classes)):
         mu, alpha = getModelParam(x, y, j)
         
         # get the final classification of the point
         # and use it to get elements of confusion matrix
         TP, TN, FP, FN = getConfMatrixParam(predictClass, y, j)
        
         # print performance parameters
         accuracy, precision, recall = getPerfomanceParam(TP, TN, FP, FN)
         fMeasure = getFMeasure(precision, recall)
        
         print("Mean of class {}: \n{}".format(j, mu))
         print("Prior class probability of class {}: {}".format(j, alpha))
         print ("Accuracy for class {}: {}".format(j, accuracy))
         print ("Precision for class {}: {}".format(j, precision))
         print ("Recall for class {}: {}".format(j, recall))
 	 print ("F-Measure for class {}: {}\n".format(j, fMeasure))   
    
    # print confusion matrix
    conf_matrix = confusion_matrix(y, predictClass, labels = classes)
    print("Confusion Matrix: \n{}\n".format(conf_matrix))
    
    # plotting confusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix)
    pl.title('Confusion matrix of nD Naive Bayes Binomial')
    fig.colorbar(cax)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show() 	 
 	 
if __name__ == "__main__":
    main()     
 