import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']

dataset = pandas.read_csv(url,names=names)

#print(dataset.groupby('class').size())
#scatter_matrix(dataset)
#print (dataset.describe())
#print dataset.shape
#print(dataset.head(20))
#dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
#dataset.hist()
#plt.show()

#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
knn_train = np.column_stack((X_train,Y_train))
knn_validation = np.column_stack((X_validation,Y_validation))


#Test options and c=evaluation metric
seed = 1
scoring = 'accuracy'

#Create Parameter Grids
hiddenLayer = [(1,2),(3,2),(6,2),(9,2),(12,2),(15,2),(20,2),(25,2),(30,2),(35,2),(40,2),(50,2),(60,2),(80,2),(100,2)]
learningRate = ['constant', 'invscaling', 'adaptive']
nn_param_grid = dict(hidden_layer_sizes=hiddenLayer,learning_rate=learningRate)

maxdepth = [1,2,4,8,10,12,15,18,25,30,35,50,65,80,100]
minsamples = [5,10,15,25,30,35,55,65,80,100]
maxfeatures = [1,2,3,4]
dt_param_grid = dict(max_depth=maxdepth,min_samples_leaf=minsamples,max_features=maxfeatures)

n_estimators = [10,20,30,40,40,50,60,80]
ada_learning_rate = [0.1,0.2,0.5,0.7,1,1.5,1.8,2,2.5,3,4,7,10]
ada_param_grid = dict(n_estimators=n_estimators,learning_rate=ada_learning_rate)

knn_neighbors = [3,5,8,10,15,20,25]
knn_weights = ['uniform','distance']
knn_param_grid = dict(n_neighbors=knn_neighbors,weights=knn_weights)

kernel = ['rbf','poly','linear']
degree = [1,2,3,4,5]
svm_C = [1,2,3,4,5,6,7,8,9,10]
svm_param_grid = dict(C=svm_C,kernel=kernel,degree=degree)



#Spot Check Alogorithms
models = []
algos = []
models.append(('KNN',KNeighborsClassifier(),knn_param_grid))
algos.append(('KNN',KNeighborsClassifier(n_neighbors=5,weights='uniform')))
models.append(('CART',DecisionTreeClassifier(),dt_param_grid))
algos.append(('CART',DecisionTreeClassifier(max_depth=10,max_features=4)))
models.append(('SVM',SVC(gamma='auto'),svm_param_grid))
algos.append(('SVM',SVC(gamma='auto',C=4,kernel='rbf')))
models.append(('NeuralNet',MLPClassifier(solver='lbfgs',activation='relu',alpha=1e-5,random_state=seed),nn_param_grid))
algos.append(('NeuralNet',MLPClassifier(solver='lbfgs',activation='relu',alpha=1e-5,random_state=seed,hidden_layer_sizes=80,learning_rate='adaptive')))
models.append(('AdaBoost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),random_state=seed),ada_param_grid))
algos.append(('AdaBoost',AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),random_state=seed,n_estimators=50,learning_rate=2)))

#Learning Curve before Hyperparameter Tuning
cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
for name, model in algos:
    title = "Learning Curves",name
    plot_learning_curve(model,title,X_train,Y_train,ylim=(0.1,1.01),cv=cv,n_jobs=4)
    plt.savefig('LC'+name+'.png')

#evaluate each model in turn
#results = []
#names = []
#for name, model,param_grid in models:
#    kfold = model_selection.KFold(n_splits=10, random_state=seed)
#    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
#    results.append(cv_results)
#    random = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv = 10, n_jobs=-1,random_state=seed)
#    start_time = time.time()
#    random_result = random.fit(X_train,Y_train)
    # Summarize results
#    print("Best: %f using %s" % (random_result.best_score_, random_result.best_params_))
#    print("Execution time: " + str((time.time() - start_time)) + ' ms')
#    names.append(name)
#    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#    print(msg)

#Find Validation Curve for the Hyperparameters found above
dt_max_depth_range = range(1,100,5)
plot_validation_curve(DecisionTreeClassifier(),"DecisionTreeClassifier_MaxDepth",X_train,Y_train,"max_depth",dt_max_depth_range,cv=10,n_jobs=-1)
plt.savefig('DecisionTree_MaxDepth1.png')
dt_max_features_range = range(1,5,1)
plot_validation_curve(DecisionTreeClassifier(),"DecisionTreeClassifier_MaxFeatures",X_train,Y_train,"max_features",dt_max_features_range,cv=10,n_jobs=-1)
plt.savefig('DecisionTree_MaxFeature')
nn_lr_range = ('constant','invscaling','adaptive')
nn_hiddenlayer_range = range(1,100,5)
plot_validation_curve(MLPClassifier(),"NeuralNet Learning Rate",X_train,Y_train,"learning_rate",nn_lr_range,cv=10,n_jobs=-1)
plt.savefig('NnLRate.png') 
plot_validation_curve(MLPClassifier(),"NeuralNet Hidden Layer",X_train,Y_train,"hidden_layer_sizes",nn_hiddenlayer_range,cv=10,n_jobs=-1)
plt.savefig('NnHL.png')
ada_nestimators_range= range(1,200,10)
ada_learningr_range = range(1,100,1)
plot_validation_curve(AdaBoostClassifier(),"AdaBoost n_estimators",X_train,Y_train,"n_estimators",ada_nestimators_range,cv=10,n_jobs=-1)
plt.savefig('Ada_nestimators.png')
plot_validation_curve(AdaBoostClassifier(),"AdaBoost Learning Rate",X_train,Y_train,"learning_rate",ada_learningr_range,cv=10,n_jobs=-1)
plt.savefig('Ada_lr.png')
knn_neighbors_range = range(1,80,2)
knn_weights_range = ('uniform','distance')
plot_validation_curve(KNeighborsClassifier(),"KNN Neighbors",X_train,Y_train,"n_neighbors",knn_neighbors_range,cv=10,n_jobs=-1)
plt.savefig('knn_neighbors.png')
plot_validation_curve(KNeighborsClassifier(),"KNN Weights",X_train,Y_train,"weights",knn_weights_range,cv=10,n_jobs=-1)
plt.savefig('knn_weights.png')
kernel_range = ['rbf','linear']
penalty_range = range(1,100,5)
plot_validation_curve(SVC(),"SVM Kernel",X_train,Y_train,"kernel",kernel,cv=10,n_jobs=-1)
plt.savefig('svm_kernel.png')
plot_validation_curve(SVC(),"SVM Penalty",X_train,Y_train,"C",penalty_range,cv=10,n_jobs=-1)
plt.show('svm_penalty.png')

#Final Prediction
for name,model in algos:
    model.fit(X_train,Y_train)
    predictions= model.predict(X_validation)
    print name,accuracy_score(Y_validation,predictions)

