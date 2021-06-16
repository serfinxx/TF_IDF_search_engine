from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_tfidf(tfidf, vocab, filename):
    # [n_doc, n_vocab]
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocab, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0] + 1), fontsize=6)
    plt.tight_layout()
    plt.savefig("./results/%s.png" % filename, format="png", dpi=500)
    plt.show()


class TfIdfSearch:
    def __init__(self, doc=(pd.read_csv("./data/train.txt", sep=';').iloc[:, 0]), q="i am so excited to see it"):
        self.q = q
        self.doc = doc
        # print(doc.head(5))

        self.vectorizer = TfidfVectorizer()

        self.tf_idf = self.vectorizer.fit_transform(self.doc)
        # print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
        # print("v2i: ", vectorizer.vocabulary_)

    def search(self):
        qtf_idf = self.vectorizer.transform([self.q])
        res = cosine_similarity(self.tf_idf, qtf_idf)
        res = res.ravel().argsort()[-3:]
        print("\ntop 3 closest sentences for '{}':\n{}".format(self.q, [self.doc[i] for i in res[::-1]]))

        i2v = {i: v for v, i in self.vectorizer.vocabulary_.items()}
        dense_tfidf = self.tf_idf.todense()

        show_tfidf(dense_tfidf, [i2v[i] for i in range(dense_tfidf.shape[1])], self.q)


if __name__ == '__main__':
    s = TfIdfSearch()
    s.search()
