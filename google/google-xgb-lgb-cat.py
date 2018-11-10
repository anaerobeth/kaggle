"""
Google Analytics Revenue Prediction
https://www.kaggle.com/c/ga-customer-revenue-prediction

Goal: predict revenue per customer of Google Merchandise Store
Part A (from previous sessions data) and Part B (for future period)
Data: preprocessed data for Part A + exported google analytics data
- https://www.kaggle.com/ogrellier/create-extracted-json-fields-dataset
- https://www.kaggle.com/satian/exported-google-analytics-data

Algorithms Used: XGB, LGB, CAT
Submissions and Public Score:

References:
- https://www.kaggle.com/zikazika/google-predictions
- https://www.kaggle.com/julian3833/2-quick-study-lgbm-xgb-and-catboost-lb-1-66
- https://www.kaggle.com/fabiendaniel/lgbm-starter
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
import gc
gc.enable()

def read_external_data(filename):
    return pd.read_csv(filename, low_memory=False, skiprows=6, dtype={"Client Id":'str'})

def merge_data(base_df, df_1, df_2):
    df = base_df.merge(pd.concat([df_1, df_2], sort=False), how="left", on="visitId")
    df.drop("Client Id", 1, inplace=True)
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
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(np.int64)

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
del train_df['totals.transactionRevenue']

