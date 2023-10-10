import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from pricing_data import krx_data
from MLutils.SklearnUtils import learning


def r_os(y_real, y_pred):  # R^2 OOS from Campbel and Thmopson
    return 1 - (y_real**2).mean() / ((y_pred.reshape(-1) - y_real)**2).mean()


r2os = make_scorer(r_os, greater_is_better=True)
rnd = 77  # random_state


class Regressions:
    def __init__(self, year, x, y, x_test, y_test):
        self.y = y.values.reshape(-1, 1)
        self.x = x.values
        self.y_test = y_test.values.reshape(-1, 1)
        self.x_test = x_test.values
        self.year = year

        self.models = ['lin', 'en', 'pls', 'pcr', 'rf', 'gbr']
        self.res_cols = ['year', 'name', 'params', 'pred_R2_OOS', 'train_score',
                         'CV_R2', 'CV_MSE', 'complexity']

        self.res = []
        self.error = []

    def execute(self, model, params, name):
        """execute Cross validating and fitting a given model.

        :param model: sklearn models
        :param params: dictionary of parameters for GridsearchCV
        :param name: name of a model
        :res: year, name, best CV result, prediction score (R^2 OOS),
          training set score(R^2), CV result(R^2), CV result(MSE), complexity
        :return: GridsearchCV result regressor
        """
        reg, _ = learning(self.x, self.y, regressor=model, params=params, pred=False, scores=[
                          'r2', 'neg_mean_squared_error'], refit='r2')

        if name == 'en':
            complexity = (reg.best_estimator_.coef_ != 0).sum()
        elif name == 'rf':
            complexity = np.mean(
                [estimator.tree_.max_depth for estimator in reg.best_estimator_.estimators_])
        elif name == 'gbr':
            complexity = np.mean(
                [estimator.max_features_ for estimator in reg.best_estimator_.estimators_.reshape(-1)])
        elif name == 'pls':
            complexity = reg.best_params_
        elif name == 'pcr':
            complexity = reg.best_params_['pca__n_components']

        pred_score = r2os(reg, self.x_test, self.y_test)

        # year, name, best CV result, prediction score (R^2 OOS),
        #   training set score(R^2), CV result(R^2), CV result(MSE), complexity
        self.res.append([self.year, name, reg.best_params_, pred_score,
                        reg.score(self.x, self.y), reg.best_score_,
                        -reg.cv_results_['mean_test_neg_mean_squared_error'], complexity])

        return reg

    @property
    def lin(self):
        # Does not use GridsearchCV
        model = LinearRegression(fit_intercept=True)
        model.fit(self.x, self.y)
        score = model.score(self.x, self.y)
        mse = mean_squared_error(self.y, model.predict(self.x))
        pred_score = r2os(model, self.x_test, self.y_test)
        self.res.append(
            [self.year, 'lin', 0, pred_score,
             model.score(self.x, self.y),
             score, mse, 0])
        return model

    @property
    def en(self):
        model = ElasticNet(max_iter=10000, random_state=rnd)
        params = {'alpha': np.linspace(
            20, 150, 10), 'l1_ratio': np.linspace(0.1, 1, 10)}
        reg = self.execute(model, params, 'en')
        return reg

    @property
    def pcr(self):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('linear', LinearRegression(fit_intercept=True))])
        params = {'pca__n_components': np.arange(1, 10)}  # self.x.shape[1]+1)}
        reg = self.execute(model, params, 'pcr')
        return reg

    @property
    def pls(self):
        model = PLSRegression(random_state=rnd)
        params = {'n_components': np.arange(1, self.x.shape[1]+1)}
        reg = self.execute(model, params, 'pls')
        return reg

    @property
    def rf(self):
        dt = DecisionTreeRegressor(random_state=rnd).fit(x, y)
        model = RandomForestRegressor(random_state=rnd)
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': np.arange(1, dt.tree_.max_depth+1, 2),
        }
        reg = self.execute(model, params, 'rf')
        return reg

    @property
    def gbr(self):
        dt = DecisionTreeRegressor(random_state=rnd).fit(x, y)
        model = GradientBoostingRegressor(random_state=rnd)
        params = {
            'learning_rate': np.linspace(0.01, 0.1, 3),
            'max_depth': [3,7,10,20, 50],
            # 'max_features': range(1, len(dt1.feature_importances_)+1)
        }
        reg = self.execute(model, params, 'gbr')
        return reg

    def fit_all(self):
        for i in self.models:
            try:
                getattr(self, i)
            except:
                print(f'Error while model {i} in {self.year}')
                self.error.append((self.year, i))

                with open('error.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.error)

        result = pd.DataFrame(self.res, columns=self.res_cols)
        if 'result.csv' in os.listdir():
            result.to_csv('result.csv', mode='a', header=False, index=False)
        else:
            result.to_csv('result.csv', index=False)

        self.result = result

    @property
    def pricing_result_(self):
        if 'result' in dir(self):
            return self.result


if __name__ == '__main__':
    train, test, test_years = krx_data()

    start = True
    pbar = tqdm(test_years)
    for year in pbar:
        pbar.set_description(f"Processing {year}")
        y, x = train[year]
        y_test, x_test = test[year]
        if start == True:
            mod = Regressions(year, x, y, x_test, y_test)
            start = False
        else:
            mod.reset(year, x, y, x_test, y_test)
        
        mod.fit_all()
