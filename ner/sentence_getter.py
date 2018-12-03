class SentenceGetter(object):

    def __init__(self, data):
        self.counter = 1
        self.data = data
        self.empty = False

    def get_next(self):
        values = [None, None, None]
        try:
            row = self.data[self.data["Sentence #"] == "Sentence: {}".format(self.counter)]
            self.counter += 1
            values = map(lambda x: row[x].values.tolist(), ["Word", "POS", "Tag"])
        except:
            self.empty = True

        return values
