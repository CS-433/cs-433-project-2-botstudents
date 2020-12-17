"""
Sklearn : 
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

#perform regularization strength grid search with cross validation on a Logistic Regression Model with CountVectorizer or TfidfVectorizer
def test_logreg(method, tweets, grid = {'logreg__C': np.logspace(-4,1,num=6)}, ngram_range = (1,1), cv = 3, stop_words=None, penalty='l2', min_df=1):
    #split dataset in train and test set for validation
    X_train, X_test, y_train, y_test = train_test_split(tweets.text,tweets.label, test_size=0.3, random_state=1)
    if method == 'bow':
        pipe = Pipeline([('vect', CountVectorizer(min_df= min_df, ngram_range=ngram_range,stop_words=stop_words)),
                    ('logreg', LogisticRegression(solver='liblinear', max_iter=1000, penalty=penalty)),
                   ])
    elif method == 'tfidf':
        pipe = Pipeline([('vect', TfidfVectorizer(min_df= min_df, ngram_range=ngram_range, stop_words=stop_words)),
                    ('logreg', LogisticRegression(solver='liblinear', max_iter=1000, penalty=penalty)),
                   ])
    else:
        print("error, wrong method")
    
    if cv > 1:
        grid_cv = GridSearchCV(pipe, grid, cv=cv, return_train_score=True, verbose=1)
        grid_cv.fit(X_train,y_train)
        print("Accuracy is " + str(100*grid_cv.score(X_test,y_test)) + "  %")
        return grid_cv
    elif cv ==1:
        pipe.fit(X_train,y_train)
        print("Accuracy is " + str(100*pipe.score(X_test,y_test)) + "  %")
        return pipe.score(X_test,y_test)

#return GridSearchCV test accuracy results plot
def plot_logreg_cv_test_score(grid_cv):
    cv_results = pd.DataFrame(grid_cv.cv_results_)
    plt.plot(cv_results['param_logreg__C'], cv_results['mean_test_score'])
    
#plot 6 different GridSearch test accuracy results in one plot
def plot_all_bow(grid_cv1, grid_cv2, grid_cv3, grid_cv4, grid_cv5, grid_cv6):
    plot_logreg_cv_test_score(grid_cv1)
    plot_logreg_cv_test_score(grid_cv2)
    plot_logreg_cv_test_score(grid_cv3)
    plot_logreg_cv_test_score(grid_cv4)
    plot_logreg_cv_test_score(grid_cv5)
    plot_logreg_cv_test_score(grid_cv6)
    plt.title('Logistic Regression : mean test scores ')
    plt.ylabel('accuracy')
    plt.xlabel('Regularization strength')
    plt.legend(['bow', 'bigram', 'trigram', '4-gram', '5-gram', 'remove stopwords'])
    plt.show()

#train logreg model for different min_df values and plot results
def test_min_df(tweets):
    X_train, X_test, y_train, y_test = train_test_split(tweets.text,tweets.label, test_size=0.3, random_state=1)
    min_dfs =[1,2,3,4,5,6,7,8,9,10]
    results = []
    for min_df in min_dfs:
        results.append(test_logreg(method='bow', tweets=tweets, ngram_range = (1,4), cv = 1, min_df=min_df))
    plt.plot(min_dfs, results)
    plt.title('Best Model Accuracy vs min_df')
    plt.ylabel('accuracy')
    plt.xlabel('min_df')
    
#perform regularization strength grid search with cross validation on a Support Vector Machine with CountVectorizer
def test_svm(tweets, kernel='linear', cv=2):
    X_train, X_test, y_train, y_test = train_test_split(tweets.text,tweets.label, test_size=0.3, random_state=1)
    
    if kernel == 'linear':
        pipe = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 4))),
            ('linearsvc', LinearSVC(max_iter=100000))
        ])
        grid = {'linearsvc__C': np.logspace(-4,1,num=6)}
    
    if kernel == 'rbf':
        pipe = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 4))),
        ('rbfsvc', SVC(kernel='rbf', gamma = 'scale'))
        ])
        grid = {'rbfsvc__C': np.logspace(-7,3,num=10)}
    
    
    grid_cv = GridSearchCV(pipe, grid, cv=cv, return_train_score=True, verbose=1)
    grid_cv.fit(X_train, y_train)
    print("Accuracy is " + str(100*grid_cv.score(X_test,y_test)) + "  %")
    
    return grid_cv