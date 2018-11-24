"""
Google Analytics Revenue Prediction
https://www.kaggle.com/c/ga-customer-revenue-prediction

Goal: predict revenue per customer of Google Merchandise Store
Part A (from previous sessions data) and Part B (for future period)
Data: preprocessed data for Part A + exported google analytics data
- https://www.kaggle.com/ogrellier/create-extracted-json-fields-dataset
- https://www.kaggle.com/satian/exported-google-analytics-data

Algorithms Used: XGB, LGB, CAT
Model Scores (CV Score):
1-LGB 1.65670
2-XGB 1.69551

References:
- https://www.kaggle.com/zikazika/google-predictions
- https://www.kaggle.com/julian3833/2-quick-study-lgbm-xgb-and-catboost-lb-1-66
- https://www.kaggle.com/fabiendaniel/lgbm-starter
"""

import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import gc
gc.enable()
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error


def read_external_data(filename):
    return pd.read_csv(filename, low_memory=False, skiprows=6, dtype={'Client Id':'str'})

def merge_data(base_df, df_1, df_2):
    df = base_df.merge(pd.concat([df_1, df_2], sort=False), how='left', on='visitId')
    df.drop('Client Id', 1, inplace=True)
    return df

# Load preprocessed data
params = {'date': str, 'fullVisitorId': str, 'sessionId':str, 'visitId': np.int64}
train = pd.read_csv('data/extracted_fields_train.csv', dtype=params)
test = pd.read_csv('data/extracted_fields_test.csv', dtype=params)

# Data from Google Analytics leak (https://www.kaggle.com/igauty/story-of-a-leak-v01)
train_store_1 = read_external_data('data/Train_external_data.csv')
train_store_2 = read_external_data('data/Train_external_data_2.csv')
test_store_1 = read_external_data('data/Test_external_data.csv')
test_store_2 = read_external_data('data/Test_external_data_2.csv')

# Get VisitId from Google Analytics
for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df['visitId'] = df['Client Id'].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)

# Merge with train/test data
train = merge_data(train, train_store_1, train_store_2)
test = merge_data(test, test_store_1, test_store_2)

print(train.columns)
print(train['Revenue'].describe())
# 903653 observations, count: 1817, unique: 929, top: $234.16

for df in [train, test]:
    df['Revenue'].fillna('$', inplace=True)
    df['Revenue'] = df['Revenue'].apply(lambda x: x.replace('$', '').replace(',', ''))
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['Revenue'].fillna(0.0, inplace=True)

target = train['totals.transactionRevenue'].fillna(0).astype(float)
target = target.apply(lambda x: np.log1p(x))
del train['totals.transactionRevenue']

# Preprocessing
# Workflow from https://www.kaggle.com/fabiendaniel/lgbm-rf-starter-lb-1-70
# 1. Drop columns with no information,
columns = [col for col in train.columns if train[col].nunique() > 1]
train = train[columns]
test = test[columns]
train_length = train.shape[0]

# Temporarily combine train and test to simplify manipulations
merged_df = pd.concat([train, test])
merged_df['diff_visitId_time'] = merged_df['visitId'] - merged_df['visitStartTime']
merged_df['diff_visitId_time'] = (merged_df['diff_visitId_time'] != 0).astype(int)

# 2. create few time-related columns
merged_df['formated_date'] = merged_df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))
merged_df['WoY'] = merged_df['formated_date'].apply(lambda x: x.isocalendar()[1])
merged_df['month'] = merged_df['formated_date'].apply(lambda x:x.month)
merged_df['quarter_month'] = merged_df['formated_date'].apply(lambda x:x.day//8)
merged_df['weekday'] = merged_df['formated_date'].apply(lambda x:x.weekday())
merged_df['formated_visitStartTime'] = merged_df['visitStartTime'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
merged_df['formated_visitStartTime'] = pd.to_datetime(merged_df['formated_visitStartTime'])
merged_df['visit_hour'] = merged_df['formated_visitStartTime'].apply(lambda x: x.hour)

# 3. label-encode the categorical columns
for col in merged_df.columns:
    if col in ['fullVisitorId', 'month', 'quarter_month', 'weekday', 'visit_hour', 'WoY']:
        continue
    if merged_df[col].dtypes == object or merged_df[col].dtypes == bool:
        merged_df[col], indexer = pd.factorize(merged_df[col])

# Delete unneeded columns
del merged_df['visitId']
del merged_df['sessionId']
del merged_df['date']
del merged_df['formated_date']
del merged_df['visitStartTime']
del merged_df['formated_visitStartTime']

numerics = [col for col in merged_df.columns if 'totals.' in col]
numerics += ['visitNumber', 'mean_hits_per_day', 'fullVisitorId']

# Check for null values
# merged_df.isnull().sum()
# Results: totals.pageviews, Sessions and Transactions
# Remove columns with null values for now
del merged_df['totals.pageviews']
del merged_df['Sessions']
del merged_df['Transactions']

categorical_feats =  [col for col in merged_df.columns if col not in numerics]

for col in categorical_feats:
    merged_df[col] = merged_df[col].astype(int)

# Once manipulations are finished, split back into train and test records
train_df = merged_df[:train_length]
test_df = merged_df[train_length:]

# Model 1 - LGBM
param = {'num_leaves': 300,
         'min_data_in_leaf': 30,
         'objective':'regression',
         'learning_rate': 0.01,
         'boosting': 'gbdt',
         'feature_fraction': 0.9,
         'metric': 'rmse',
         'verbosity': -1}

train_cols = [col for col in train_df.columns if col not in ['fullVisitorId']]
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
features = list(train_df[train_cols].columns)
feature_importance_df = pd.DataFrame()
folds_split = folds.split(train_df.values, target.values)

for fold_, (train_index, val_index) in enumerate(folds_split):
    train_data = lgb.Dataset(train_df.iloc[train_index][train_cols], label=target.iloc[train_index], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_index][train_cols], label=target.iloc[val_index], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, train_data, num_round, valid_sets = [train_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_index] = clf.predict(train_df.iloc[val_index][train_cols], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df['feature'] = features
    fold_importance_df['importance'] = clf.feature_importance()
    fold_importance_df['fold'] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(test_df[train_cols], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
# CV score: 1.65670
# Score is worse compared to 1.61782 score in the source kernel
# Possible reasons: differences in tuning params used;
# removed feature with null values like 'totals.pageviews' may be important

cols = feature_importance_df[['feature', 'importance']].groupby('feature').mean().sort_values(
    by='importance', ascending=False)[:1000].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

print(best_features.sort_values(by="importance", ascending=False).head(20))
# Features by decreasing importance: total.hits, WoY, visitNumber, visit_hour
# quarter_month, geoNetwork.city, weekday

# Create submission
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = np.expm1(predictions)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test['PredictedLogRevenue'] = np.log1p(grouped_test['PredictedLogRevenue'])
grouped_test.to_csv('submit.csv',index=False)


# Model 2 - XGBoost
param = {
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'eta': 0.001,
    'max_depth': 10,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'alpha':0.001,
    'random_state': 42,
    'silent': True
}

# DMatrix requirement: DataFrame.dtypes for data must be int, float or bool.
# Use label-encoded data instead of raw df valuesa but delete 'fullVisitorId'
del train_df['fullVisitorId']

X_train, X_val, y_train, y_val = train_test_split(train_df, target, test_size=0.15, random_state=1)
xgb_train = xgb.DMatrix(X_train, y_train)
xgb_val = xgb.DMatrix(X_val, y_val)

model = xgb.train(
    param,
    xgb_train,
    num_boost_round=2000,
    evals= [(xgb_train, 'train'), (xgb_val, 'valid')],
    early_stopping_rounds=100,
    verbose_eval=500
)

y_pred_train = model.predict(xgb_train, ntree_limit=model.best_ntree_limit)
y_pred_val = model.predict(xgb_val, ntree_limit=model.best_ntree_limit)
print(f"XGB : RMSE val: {rmse(y_val, y_pred_val)}  - RMSE train: {rmse(y_train, y_pred_train)}")
# [1999] train-rmse:1.6204 valid-rmse:1.69551
