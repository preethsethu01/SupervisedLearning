import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.svm import SVC
import random
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

def plot_validation_curve(estimator,title,X,y,param_name,param_range,cv=None,
                        n_jobs=None):
    plt.figure()
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy Score")
    train_scores, test_scores = validation_curve(
        estimator, X, y,param_name=param_name,param_range=param_range, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="blue")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(param_range, train_scores_mean, 'o-', color="blue",
             label="Training score")
    plt.plot(param_range, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#Load Data Set
filename = 'breast-cancer-wisconsin.data.csv'
names = ['Id','ClumpThickness','CellSize','CellShape','MAdhesion','SingleECellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','class']
data = pandas.read_csv(filename,names=names)
to_drop = ['?']
df = data[~data['BareNuclei'].isin(to_drop)]
df = df.astype('int')


array = df.values
X = array[:100,0:10]
Y = array[:100,10]
validation_size = 0.20
seed  = 7
X_train,X_validation,Y_train,Y_validation = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)

knn_train = np.column_stack((X_train,Y_train))
knn_test = np.column_stack((X_validation,Y_validation))

kernel_range = ['rbf','linear']
penalty_range = range(1,100,5)
plot_validation_curve(SVC(),"SVM Kernel",X_train,Y_train,"kernel",kernel_range,cv=10,n_jobs=-1)
plt.savefig('svm_kernel.png')
plot_validation_curve(SVC(),"SVM Penalty",X_train,Y_train,"C",penalty_range,cv=10,n_jobs=-1)
plt.savefig('svm_penalty.png')
