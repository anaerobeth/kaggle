"""
Santander Value Prediction Challenge
https://www.kaggle.com/c/santander-value-prediction-challenge

Goal: identify the value of transactions for each potential customer
Data: anonymized dataset containing numeric feature variables
https://www.kaggle.com/c/santander-value-prediction-challenge/data

Algorithms Used: XGB with feature scoring, LGB
Submissions and Public Score:
1-XGB+LGB+with leak train data - 4.55
2-XGB+LGB+tuning+with leak train data - 1.56943

References:
- https://www.kaggle.com/zeus75/xgboost-features-scoring-with-ligthgbm-model
- https://www.kaggle.com/johnfarrell/breaking-lb-fresh-start-with-lag-selection
- https://www.kaggle.com/ogrellier/feature-scoring-vs-zeros
- https://www.kaggle.com/tezdhar/breaking-lb-fresh-start
- https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import KFold

print('loading data')
data = pd.read_csv('data/train.csv')
target = np.log1p(data['target'])
data.drop(['ID', 'target'], axis=1, inplace=True)
test = pd.read_csv('data/test.csv')

# Add leaked training data from Kernel: Breaking LB - Fresh start with Lag Selection
# https://www.kaggle.com/johnfarrell/breaking-lb-fresh-start-with-lag-selection/output
# Exploits the fact that dataset is a time series in both dimensions
leak = pd.read_csv('data/train_leak.csv')
data['leak'] = leak['compiled_leak'].values
data['log_leak'] = np.log1p(leak['compiled_leak'].values)

# Feature scoring with XGB
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** .5

reg = XGBRegressor(n_estimators=10)
folds = KFold(n_splits=4, shuffle=True, random_state=42)
fold_index = [(train, val) for train, val in folds.split(data)]
scores = []

nb_values = data.nunique(dropna=False)
nb_zeros = (data == 0).astype(np.uint8).sum(axis=0)

nonfeature_cols =  ['log_leak', 'leak', 'target', 'ID']
features = [f for f in data.columns if f not in nonfeature_cols]

print('running xgb')
for feature in features:
    score = 0
    for train, val in fold_index:
        reg.fit(
            data[['log_leak', feature]].iloc[train],
            target.iloc[train],
            eval_set=[(data[['log_leak', feature]].iloc[val], target.iloc[val])],
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=False
        )
        score += rmse(
            target.iloc[val],
            reg.predict(
                data[['log_leak', feature]].iloc[val],
                ntree_limit=reg.best_ntree_limit)
            ) / folds.n_splits
    scores.append((feature, score))

report = pd.DataFrame(scores, columns=['feature', 'rmse']).set_index('feature')
report['nb_zeros'] = nb_zeros
report['nunique'] = nb_values
report.sort_values(by='rmse', ascending=True, inplace=True)

report.to_csv('feature_report.csv', index=True)

report  = pd.read_csv('feature_report.csv')

print('feature selection')
# Feature selection
low_rmse = report['rmse'] <= 0.7955
good_features = report.loc[low_rmse].index
rmses = report.loc[low_rmse, 'rmse'].values

# Add leak to test
test_leak = pd.read_csv('data/test_leak.csv')
test['leak'] = test_leak['compiled_leak']
test['log_leak'] = np.log1p(test_leak['compiled_leak'])

# Model 1
# Lightgbm

# Use 5 splits this time
folds = KFold(n_splits=5, shuffle=True, random_state=1)
def add_stats(df):
    df['log_of_mean'] = np.log1p(df[features].replace(0, np.nan).mean(axis=1))
    df['mean_of_log'] = np.log1p(df[features]).replace(0, np.nan).mean(axis=1)
    df['log_of_median'] = np.log1p(df[features].replace(0, np.nan).median(axis=1))
    df['nb_nans'] = df[features].isnull().sum(axis=1)
    df['the_sum'] = np.log1p(df[features].sum(axis=1))
    df['the_std'] = df[features].std(axis=1)
    df['the_kur'] = df[features].kurtosis(axis=1)

    return df

data.replace(0, np.nan, inplace=True)
data = add_stats(data)

test.replace(0, np.nan, inplace=True)
test = add_stats(test)

# Only use good features, log leak and stats for training
extra_features = ['log_leak', 'log_of_mean', 'mean_of_log', 'log_of_median', 'nb_nans', 'the_sum', 'the_std', 'the_kur']
final_features = good_features.tolist() + extra_features
oof_preds = np.zeros(data.shape[0])
test['target'] = 0

# Params from zeus75 Kaggle kernel referenced above
lgb_params = {
    'objective': 'regression',
    'num_leaves': 58,
    'subsample': 0.6143,
    'colsample_bytree': 0.6453,
    'min_split_gain': np.power(10, -2.5988),
    'reg_alpha': np.power(10, -2.2887),
    'reg_lambda': np.power(10, 1.7570),
    'min_child_weight': np.power(10, -0.1477),
    'verbose': -1,
    'seed': 3,
    'boosting_type': 'gbdt',
    'max_depth': -1,
    'learning_rate': 0.05,
    'metric': 'l2',
}

print('running lgb')
def run_lgb(data, lgb, dtrain, target, off_preds, lgb_params):
    for train, val in folds.split(data):
        clf = lgb.train(
            params=lgb_params,
            train_set=dtrain.subset(train),
            valid_sets=dtrain.subset(val),
            num_boost_round=10000,
            early_stopping_rounds=100,
            verbose_eval=0
        )
        oof_preds[val] = clf.predict(dtrain.data.iloc[val])
        test['target'] += clf.predict(test[features]) / folds.n_splits
        print(mean_squared_error(target.iloc[val], oof_preds[val]) ** .5)

dtrain = lgb.Dataset(data=data[features], label=target, free_raw_data=False)
dtrain.construct()

run_lgb(data, lgb, dtrain, target, oof_preds, lgb_params)

data['predictions'] = oof_preds
data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(),'leak'])
print('OOF SCORE : %9.6f' % (mean_squared_error(target, oof_preds) ** .5))
# 12.66 on folds with n_split=4
print('OOF SCORE with LEAK : %9.6f' % (mean_squared_error(target, data['predictions']) ** .5))
# 5.44 on folds with n_split=4

test['target'] = np.expm1(test['target'])
test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
test[['ID', 'target']].to_csv('xgb-lgb-leak1.csv', index=False, float_format='%.2f')

# Model 2
# Tune lgb params
lgb_params2 = {
    'objective': 'regression',
    'n_estimators': 500,
    'max_bin': 10,
    'subsample': 0.8,
    'subsample_freq': 10,
    'colsample_bytree': 0.8,
    'learning_rate': 0.02,
    'min_child_samples': 500
}

run_lgb(data, lgb, dtrain, target, oof_preds, lgb_params2)
# 1.40596 on folds with n_split=4
# 1.39831 on folds with n_split=5

data['predictions'] = oof_preds
data.loc[data['leak'].notnull(), 'predictions'] = np.log1p(data.loc[data['leak'].notnull(),'leak'])
print('OOF SCORE : %9.6f' % (mean_squared_error(target, oof_preds) ** .5))
# 1.422552 on folds with n_split=4
# 1.428494 on folds with n_split=5

print('OOF SCORE with LEAK : %9.6f' % (mean_squared_error(target, data['predictions']) ** .5))
# 0.715947 on folds with n_split=4
# 0.716773 on folds with n_split=5

test['target'] = np.expm1(test['target'])
test.loc[test['leak'].notnull(), 'target'] = test.loc[test['leak'].notnull(), 'leak']
test[['ID', 'target']].to_csv('xgb-lgb-leak2.csv', index=False, float_format='%.2f')
