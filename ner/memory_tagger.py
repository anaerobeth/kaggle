from sklearn.base import BaseEstimator, TransformerMixin

class MemoryTagger(BaseEstimator, TransformerMixin):
    """Adapted from https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/"""

    def __init__(self):
        self.voc = {}
        self.memory = {}
        self.tags = []

    def fit(self, X, y):
        """ Expects a list of words as X and a list of tags as y"""
        for x, t in zip(X, y):
            if t not in self.tags:
                self.tags.append(t)
            if x in self.voc:
                self.voc[x][t] = (self.voc[x][t] + 1) if t in self.voc[x] else 1
            else:
                self.voc[x] = {t: 1}

        for k, d in self.voc.items():
            self.memory[k] = max(d, key=d.get)

    def predict(self, X, y=None):
        """Predict the the tag from memory. If word is unknown, predict 'O'."""
        return [self.memory.get(x, 'O') for x in X]
