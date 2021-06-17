from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd


class TfIdfSearch:
    def __init__(self, doc, q):
        self.q = q
        self.doc = doc
        # print(doc.head(5))

        # calculate tf-idf of texts
        self.vectorizer = TfidfVectorizer()
        self.tf_idf = self.vectorizer.fit_transform(self.doc)
        # print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
        # print("v2i: ", vectorizer.vocabulary_)

    def search(self, start_index, end_index):
        qtf_idf = self.vectorizer.transform([self.q])
        res = cosine_similarity(self.tf_idf, qtf_idf)
        selected = res.ravel().argsort()[::-1][start_index:end_index]
        # print("\n{}th to {}th closest documents for '{}':\n{}".format(start_index, end_index, self.q, [self.doc[i]
        # for i in selected]))

        # v2i = {v: i for v, i in self.vectorizer.vocabulary_.items()}

        return [self.doc[i] for i in selected]


if __name__ == '__main__':
    s = TfIdfSearch(doc=(pd.read_csv("../data/train.txt", sep=';').iloc[:, 0]), q="i am so excited to see it")
    s.search(0, 3)
