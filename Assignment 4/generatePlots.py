# Author: Parikshita Tripathi
# Time : 04/07/2016

#--------------------------------------------------------------------------#

# import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve 

#---------------------------------------------------------------------
# function to plot linear data
def plotMargin(X1_train, X2_train, w, w0, sv):

    # store margin values in a variable
    ax = [0, 1, -1]

    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    plt.scatter(sv[:,0], sv[:,1], s=100, c="g")
    
    for i in ax: 
        a0 = 4; b0 = -4
        cond1 = -w[0] * a0 - w0
        cond2 = -w[0] * b0 - w0
        # w.x + b = 0
        if i == 0:  
            a1 = (cond1)/w[1]
            b1 = (cond2)/w[1]
            plt.plot([a0,b0], [a1,b1], "k")
               
        # w.x + b = 1
        elif i == 1:    
            a1 = (cond1 + i)/w[1]
            b1 = (cond2 + i)/w[1]
            plt.plot([a0,b0], [a1,b1], "k--")
                
        # w.x + b = -1
        elif i == -1:
            a1 = (cond1 + i)/w[1]
            b1 = (cond2 + i)/w[1]
            plt.plot([a0,b0], [a1,b1], "k--")
        plt.show()

#---------------------------------------------------------------------    
# function to plot confusion matrix
def plotConfusionMatrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


---------------------------------------------------------------------
# function to generate plot of datasets    
def genPlot(X, Y):
    return plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y)

   
#---------------------------------------------------------------------
# function to generate precision recall curve
def plotPrecisionRecall(y_test, y_predict):
    precision = dict()
    recall = dict()
    average_precision = dict()
    classes = np.unique(y_test)
    for i, j in enumerate(classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[np.ix_(y_test == j)], y_predict[np.ix_(y_test == j)])
        average_precision[i] = average_precision_score(y_test[np.ix_(y_test == j)], y_predict[np.ix_(y_test == j)])
   
    plt.plot(recall[0], precision[0], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()


