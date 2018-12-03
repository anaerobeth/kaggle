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

from sentence_getter import SentenceGetter

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
new_classes = data['Tag'].unique().copy()
list(new_classes).pop() # O tag is the last class

getter = SentenceGetter(data)
sent, pos, tag = getter.get_next()

print(sent[:8])
print(pos[:8])
print(tag[:8])
# ['Thousands', 'of', 'demonstrators', 'have', 'marched', 'through', 'London', 'to']
# ['NNS', 'IN', 'NNS', 'VBP', 'VBN', 'IN', 'NNP', 'TO']
# ['O', 'O', 'O', 'O', 'O', 'O', 'B-geo', 'O']

