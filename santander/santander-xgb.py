"""
Santander Value Prediction Challenge
https://www.kaggle.com/c/santander-value-prediction-challenge

Goal: identify the value of transactions for each potential customer
Data: anonymized dataset containing numeric feature variables
https://www.kaggle.com/c/santander-value-prediction-challenge/data

Algorithms Used: XGB, CatBoostRegressor
Submissions and Public Score:
1-XGB-1 - 2.08572
3-Cat-1 - 11.9282

References:
- https://www.kaggle.com/samratp/santander-value-prediction-xgb-and-lightgbm
- https://tech.yandex.com/catboost/doc/dg/concepts/python-usages-examples-docpage/
"""

import pdb
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split

print('done with imports')
def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'eta': 0.005,
          'max_depth': 15,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'alpha':0,
          'random_state': 42,
          'silent': True}

    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)

    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=100)

    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))

    return xgb_pred_y, model_xgb

print('importing data')
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Count null values in columns
# null_columns=train.columns[train.isnull().any()]
# train[null_columns].isnull().sum()
# Series([], dtype: float64)

# test_null_columns=test.columns[test.isnull().any()]
# test[test_null_columns].isnull().sum()
# Series([], dtype: float64)

# Remove non-informative columns from train (same value for all rows)
train_df = train.loc[:, (train != train.iloc[0]).any()]

# Ensure test has same columns as train (avoid feature mismatch when running XGB)
test_df = test[train_df.columns.drop('target')]

# # Check range of values
# df = train_df.drop('ID', axis=1)
# df = df.drop('target', axis=1)
# diff = df.max() - df.min()
# # Result: Ranges are from 0 to 10^9, need to scale values

# # Divide values in each column by the max() value of that column
# for col in diff.index.tolist():
#     train_df.loc[:, col] = train_df[col] / (diff[col] + 1)
#     # Add 1 avoid zero division error

print('creating model')
# XGB model
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)
X_test = test_df.drop(["ID"], axis=1)

dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)

print('making predictions')
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")

sub = pd.read_csv('data/sample_submission.csv')
sub["target"] = pred_test_xgb

print(sub.head())
sub.to_csv('xgb-1.csv', index=False)


# Model 2
# Use Scaler to normalize values
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
print('making predictions using scaled values')
pred_test_xgb2, model_xgb2 = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training on Scaled Values Completed...")
# Result the same as model 1: [1438] train-rmse:0.719389 valid-rmse:1.42394

# Model 3
# CatBoostRegressor - gradient boosting on decision trees by Yandex
from catboost import CatBoostRegressor
model_cb = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=4, eval_metric='RMSE')
model_cb.fit(X_train, y_train, cat_features=[], use_best_model=True)
preds = model_cb.predict(X_test)
# Result: 999: learn: 1.1991732 total: 1m 58
# Scored worse on the Kaggle dataset


