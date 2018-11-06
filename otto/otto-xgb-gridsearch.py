"""
Otto Group Product Classification Challenge
https://www.kaggle.com/c/otto-group-product-classification-challenge

Goal: distinguish between 10 main product categories
Data: dataset with 93 features for more than 200,000 products
https://www.kaggle.com/c/otto-group-product-classification-challenge/data

Algorithms Used: XGB
This code identifies the best parameters using GridSearch

Reference:
https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
"""
import pickle
from pandas import read_csv, to_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

data = read_csv('data/train.csv')
dataset = data.values
X = dataset[:,0:94]
y = dataset[:,94]
label_encoded_y = LabelEncoder().fit_transform(y)

# Grid search
model = XGBClassifier()
n_estimators = range(50, 400, 50)
param_grid = dict(n_estimators=n_estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, label_encoded_y)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score'] * -1
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param)

# Best: -0.000848 using {'n_estimators': 250}

xgb = grid_search.best_estimator_
""" xgb is the classifier
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=250,
       n_jobs=1, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
"""
xgb.fit(X, label_encoded_y)
preds = xgb.predict(test)
pickle.dump(xgb, open("xgb.pickle.dat", "wb"))
