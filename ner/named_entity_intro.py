"""
Find the entity-type of words

Data: extract from GMB corpus, tagged and annotated for named entities 
https://www.kaggle.com/abhinavwalia95/how-to-loading-and-fitting-dataset-to-scikit/data

Algorithms Used: RandomForest
Scores:

References:
- https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/
- https://www.kdnuggets.com/2018/10/named-entity-recognition-classification-scikit-learn.html
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

from sentence_getter import SentenceGetter
from memory_tagger import MemoryTagger

data = pd.read_csv("data/ner_dataset.csv", encoding="latin1")
# Replace NaN with the preceding non-NaN value
data = data.fillna(method="ffill")

print(data['Sentence #'].values[-1]) # 'Sentence: 47959'
print(len(data['Word']), len(data['Word'].unique())) # 1048575 35178
print(len(data['Tag'].unique())) # 17

print(data.groupby('Tag').size())
print(sorted(data.groupby('Tag').size()))
# [51, 198, 201, 253, 297, 308, 402, 6528, 7414, 15870, 16784, 16990, 17251, 20143, 20333, 37644, 887908]
# O tag count is 887908 (almost 85% of labels)

# When evaluating classification metrics, remove O tag since it dominates the data set
new_classes = data['Tag'].unique().tolist().copy()
new_classes.pop(0) # O tag is the first class

getter = SentenceGetter(data)
sent, pos, tag = getter.get_next()

print(sent[:8])
print(pos[:8])
print(tag[:8])
# ['Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to']
# ['NNS', 'IN', 'NNS', 'VBP', 'VBN', 'IN', 'NNP', 'TO']
# ['O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O']


# Model 1 - Baseline
# Memorize the training data and predict the most common entity for a given word
tagger = MemoryTagger()
tagger.fit(sent, tag)

words = data["Word"].values.tolist()
tags = data["Tag"].values.tolist()
pred = cross_val_predict(estimator=MemoryTagger(), X=words, y=tags, cv=5)
report = classification_report(y_pred=pred, y_true=tags, labels=new_classes)
# Report        precision    recall  f1-score   support
# weighted avg       0.76      0.68      0.71    160667
