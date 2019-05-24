import warnings
warnings.filterwarnings("ignore")
from getEmbeddings import getEmbeddings
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
import pickle

def plot_cmat(yte, ypred):
    # confusion matrix
    warnings.filterwarnings("ignore")
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

def develop():
    
    # Read the data
    if not os.path.isfile('./xtr.npy') or \
        not os.path.isfile('./xte.npy') or \
        not os.path.isfile('./ytr.npy') or \
        not os.path.isfile('./yte.npy'):
        xtr,xte,ytr,yte = getEmbeddings("datasets/train.csv")
        np.save('./xtr', xtr)
        np.save('./xte', xte)
        np.save('./ytr', ytr)
        np.save('./yte', yte)

    xtr = np.load('./xtr.npy')
    xte = np.load('./xte.npy')
    ytr = np.load('./ytr.npy')
    yte = np.load('./yte.npy')
    print("Here")
    # Use the built-in SVM for classification
    clf = SVC()
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)
    m = yte.shape[0]
    n = (yte != y_pred).sum()
    print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

    filename = 'finalized_model.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    print("Classified")

    # Draw the confusion matrix
    plot_cmat(yte, y_pred)



