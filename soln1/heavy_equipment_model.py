from sklearn.base import TransformerMixin
from sklearn.cross_validation import PredefinedSplit
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer


class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])


class FilterColumns(CustomMixin):
    def fit(self, X, y):
        column_counts = X.apply(lambda x: x.count(), axis=0)
        self.keep_columns = column_counts[column_counts == column_counts.max()]
        return self

    def transform(self, X):
        return X.ix[:, self.keep_columns.index]


class ReplaceOutliers(CustomMixin):
    def fit(self, X, y):
        self.replace_value = X.YearMade[X.YearMade > 1900].mode()
        return self

    def transform(self, X):
        condition = X.YearMade > 1900
        X['YearMade_imputed'] = 0
        X.ix[~condition, 'YearMade'] = self.replace_value[0]
        X.ix[~condition, 'YearMade_imputed'] = 1
        return X


class ComputeAge(CustomMixin):
    '''Compute the age of the vehicle at sale.
    '''
    def fit(self, X, y):
        return self

    def transform(self, X):
        saledate = pd.to_datetime(X.saledate)
        X['equipment_age'] = saledate.dt.year - X.YearMade
        return X


class ComputeNearestMean(CustomMixin):
    '''Compute a mean price for similar vehicles.
    '''
    def __init__(self, window=5):
        self.window = window

    def get_params(self, **kwargs):
        return {'window': self.window}

    def fit(self, X, y):
        X = X.sort_values(by=['saledate_converted'])
        g = X.groupby('ModelID')['SalePrice']
        m = g.apply(lambda x: x.rolling(self.window).agg([np.mean]))

        ids = X[['saledate_converted', 'ModelID', 'SalesID']]
        z = pd.concat([m, ids], axis=1)
        z['saledate_converted'] = z.saledate_converted + timedelta(1)
        # Some days will have more than 1 transaction for a particular model,
        # take the last mean (which has most info)
        z = z.drop('SalesID', axis=1)
        groups = ['ModelID', 'saledate_converted']
        self.averages = z.groupby(groups).apply(lambda x: x.tail(1))
        # This is kinda unsatisfactory, but at least ensures
        # we can always make predictions
        self.default_mean = X.SalePrice.mean()
        return self

    def transform(self, X):
        near_price = pd.merge(self.averages, X, how='outer',
                              on=['ModelID', 'saledate_converted'])
        nxcols = ['ModelID', 'saledate_converted']
        near_price = near_price.set_index(nxcols).sort_index()
        g = near_price['mean'].groupby(level=0)
        filled_means = g.transform(lambda x: x.fillna(method='ffill'))
        near_price['filled_mean_price'] = filled_means
        near_price = near_price[near_price['SalesID'].notnull()]
        missing_mean = near_price.filled_mean_price.isnull()
        near_price['no_recent_transactions'] = missing_mean
        near_price['filled_mean_price'].fillna(self.default_mean, inplace=True)
        return near_price


class DataType(CustomMixin):
    col_types = {'str': ['MachineID',
                         'ModelID', 'datasource']}

    def fit(self, X, y):
        return self

    def transform(self, X):
        for type, columns in self.col_types.iteritems():
            X[columns] = X[columns].astype(type)
        X['saledate_converted'] = pd.to_datetime(X.saledate)
        return X


class ColumnFilter(CustomMixin):
    columns = ['YearMade', 'YearMade_imputed', 'equipment_age',
               'filled_mean_price', 'no_recent_transactions']

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        X = X.set_index('SalesID')[self.columns].sort_index()
        return X


if __name__ == '__main__':
    df = pd.read_csv('Train.csv')
    df = df.set_index('SalesID').sort_index()
    y = df.SalePrice

    # This is for predefined split... we want -1 for our training split,
    # 0 for the test split.
    cv_cutoff_date = pd.to_datetime('2011-01-01')
    cv = -1*(pd.to_datetime(df.saledate) < cv_cutoff_date).astype(int)

    cross_val = PredefinedSplit(cv)

    p = Pipeline([
        ('filter', FilterColumns()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('lm', LinearRegression())
    ])
    df = df.reset_index()

    def rmsle(y_hat, y):
        target = y
        predictions = y_hat
        log_diff = np.log(predictions+1) - np.log(target+1)
        return np.sqrt(np.mean(log_diff**2))

    # GridSearch
    params = {'nearest_average__window': [3, 5, 7]}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, params,
                        scoring=rmsle_scorer,
                        cv=cross_val)
    clf = gscv.fit(df.reset_index(), y)

    print 'Best parameters: %s' % clf.best_params_
    print 'Best RMSLE: %s' % clf.best_score_

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)
