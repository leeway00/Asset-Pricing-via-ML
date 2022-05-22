import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from MLutils.BaseUtils import *

import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, confusion_matrix, classification_report, \
    accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate, \
    RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline





def evaluate(model, X, y, cv, 
             scoring = ['r2', 'neg_mean_squared_error'],
             validate = True):
    if validate:
        verbose = 0
    else:
        verbose = 2
    scoring = ['r2', 'neg_mean_squared_error']
    cv_results = cross_validate(model, X, y, cv=cv,
        scoring = scoring, verbose=verbose)
    return cv_results

################ This should be adjusted ###################
def regress_test(X, y, regressor, params = None,
            target ='daily_ret', window = 120, pred_window = 30):
    # training with 6month(120days) and predict 3month(60days)
    tscv = TimeSeriesSplit() # n_splits=_num_batch

    pf = PolynomialFeatures(degree=1)
    alphas = np.geomspace(50, 800, 20)
    scores=[]
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, max_iter=100000)

        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ("polynomial_features", pf),
            ("ridge_regression", ridge)])

        r2, mse = evaluate(estimator, X, y, cv = tscv)
        scores.append(np.mean(r2))
    plt.plot(alphas, scores)



def measure_error(y_train, y_test, pred_train, pred_test, label):
    train =  pd.Series({'accuracy':accuracy_score(y_train, pred_train),
                    'precision': precision_score(y_train, pred_train),
                    'recall': recall_score(y_train, pred_train),
                    'f1': f1_score(y_train, pred_train)},
                    name='train')
    
    test = pd.Series({'accuracy':accuracy_score(y_test, pred_test),
                    'precision': precision_score(y_test, pred_test),
                    'recall': recall_score(y_test, pred_test),
                    'f1': f1_score(y_test, pred_test)},
                    name='test')

    return pd.concat([train, test], axis=1)



def confusion_plot(y_true, y_pred):
    _, ax = plt.subplots(figsize=None)
    ax = sns.heatmap(confusion_matrix(y_true, y_pred), 
                    annot=True, fmt='d',
                    annot_kws={"size": 40, "weight": "bold"})
    labels = ['False', 'True']
    ax.set_xticklabels(labels, fontsize=25);
    ax.set_yticklabels(labels, fontsize=25);
    ax.set_ylabel('True value', fontsize=30);
    ax.set_xlabel('Prediction', fontsize=30)
    return ax


def class_report(y_true, y_pred):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)






def execute_CV(model, param_grid, X, y, cv, poly = None, gridsearch = True, **kwargs):
    if poly != None:
        # when both polynomial features and parameter grid are used
        scores = {}
        poly_able = (X.dtypes != 'uint8').values
        X_poly, X_non = X.iloc[:, poly_able], X.iloc[:, ~poly_able]
        for i in tqdm(poly):
            X2 = PolynomialFeatures(degree=i).fit_transform(X_poly)
            X2 = np.concatenate([X2, X_non], axis=1)
            if gridsearch:
                CV_ = GridSearchCV(model, param_grid, cv=cv, **kwargs)
            else:
                CV_ = RandomizedSearchCV(model, param_grid, cv=cv, **kwargs)
            CV_.fit(X2, y)
            scores[CV_.best_score_] = (i, CV_)
            
        mxx = scores[max(scores.keys())]
        print(mxx[0])
        return mxx[1]
    
    else:
        # When only parameter grid are used
        if gridsearch:
            CV_ =  GridSearchCV(model, param_grid, cv=cv, **kwargs)
        else:
            CV_ = RandomizedSearchCV(model, param_grid, cv=cv, **kwargs)
        CV_.fit(X,y)
        print('Best score:', CV_.best_score_)
        return CV_


def learning(X: pd.DataFrame, y, regressor, params = None, clss = False, pred = False,
        n_jobs = None, poly = None, scores = None, Date = 'Date', gridsearch = True,
        window = 400, pred_window = 15, prnt = True, refit = True, verbose = 1):

    # training with 6month(120days) and predict 3month(60days)
    if pred == True:
        X, X_pred = train_test_split(X, test_size=0.1, shuffle = False)
        y, y_pred = train_test_split(y, test_size=0.1, shuffle = False)

    tscv = TimeSeriesSplit() #n_splits=int(data.shape[0]), max_train_size=window
    
    if params != None:
        cvres =  execute_CV(regressor, params, X, y, tscv, poly = poly, gridsearch = gridsearch,
                                scoring= scores, n_jobs = n_jobs, refit = refit, verbose = verbose)
        if pred:
            if prnt:
                if clss != True:
                    print(r2_score(y_pred, cvres.predict(X_pred)))
                print(confusion_plot(y_pred>0, cvres.predict(X_pred)>0))
            rpt = class_report(y_pred, cvres.predict(X_pred))
            return cvres, rpt
        else:
            return cvres, None
    else:
        # cross validation only with polynomial features
        if poly != None:
            scores = {}
            poly_able = (X.dtypes != 'uint8').values
            X_poly, X_non = X.iloc[:, poly_able], X.iloc[:, ~poly_able]
            for i in tqdm(poly):
                X2 = PolynomialFeatures(degree=i).fit_transform(X_poly)
                X2 = np.concatenate([X2, X_non], axis=1)
                cv_results = cross_validate(regressor, X2, y, cv = tscv,
                                            verbose=verbose)
                scores[i] = cv_results
                if prnt:
                    print(scores)
            return regressor.fit(X2, y), scores
        else:
            # no cross validation
            res = []
            reg = regressor.fit(X, y)
            if pred:
                if prnt:
                    if clss != True:
                        res.append(r2_score(y_pred, reg.predict(X_pred)))
                        print(confusion_plot(y_pred>0, reg.predict(X_pred)>0))
                    else:
                        res.append(class_report(y_pred, reg.predict(X_pred)))
                    print(confusion_plot(y_pred>0, reg.predict(X_pred)>0))
            else:
                res = evaluate(reg, X, y, tscv, clss)
            return reg, res
            
    
            