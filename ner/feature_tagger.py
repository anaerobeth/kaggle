from sklearn.preprocessing import LabelEncoder

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Adapted from https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/"""

    def __init__(self):
        self.memory_tagger = MemoryTagger()
        self.tag_encoder = LabelEncoder()
        self.pos_encoder = LabelEncoder()

    def fit(self, X, y):
        words, self.pos, tags = map(lambda col: X[col].values.tolist(), ["Word", "POS", "Tag"])
        self.memory_tagger.fit(words, tags)
        self.tag_encoder.fit(tags)
        self.pos_encoder.fit(self.pos)
        return self

    def pos_default(self, p):
        return self.pos_encoder.transform([p])[0] if p in self.pos else -1

    def transform_tag(self, tag):
        return self.tag_encoder.transform(tag)[0]

    def set_defaults():
        w = transform_tag['O'])
        pos = pos_default(".")
        return w, pos

    def predict_tag(self, word):
        return self.memory_tagger.predict(word)

    def transform(self, X, y=None):
        pos, words = map(lambda col: X[col].values.tolist(), ["POS", "Word"])
        out = []
        for i in range(len(words)):
            w = words[i]
            p = pos[i]
            if i < len(words) - 1:
                wp = transform_tag(predict_tag([words[i+1]]))
                posp = pos_default(pos[i+1])
            else:
                wp, posp = set_defaults()
            if i > 0:
                if words[i-1] != ".":
                    wm = transform_tag(predict_tag([words[i-1]]))
                    posm = pos_default(pos[i-1])
                else:
                    wm, posm = set_defaults()
            else:
                wm, posm = set_defaults()
            out.append(np.array([w.istitle(), w.islower(), w.isupper(), len(w), w.isdigit(), w.isalpha(),
                                 transform_tag(predict_tag([w])), pos_default(p), wp, wm, posp, posm]))
        return out
