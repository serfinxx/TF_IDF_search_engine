from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wordcloud import WordCloud

import pandas as pd
import matplotlib.pyplot as plt


class TfIdfSearch:
    def __init__(self, doc=(pd.read_csv("./data/train.txt", sep=';').iloc[:, 0]), q="i am so excited to see it"):
        self.q = q
        self.doc = doc
        # print(doc.head(5))

        # calculate tf-idf of texts
        self.vectorizer = TfidfVectorizer()
        self.tf_idf = self.vectorizer.fit_transform(self.doc)
        # print("idf: ", [(n, idf) for idf, n in zip(vectorizer.idf_, vectorizer.get_feature_names())])
        # print("v2i: ", vectorizer.vocabulary_)

    def search(self):
        qtf_idf = self.vectorizer.transform([self.q])
        res = cosine_similarity(self.tf_idf, qtf_idf)
        top3res = res.ravel().argsort()[-3:]
        print("\ntop 3 closest sentences for '{}':\n{}".format(self.q, [self.doc[i] for i in top3res[::-1]]))

        i2v = {i: v for v, i in self.vectorizer.vocabulary_.items()}

        # plot word cloud
        zip_iterator = zip(self.doc, sum(res.tolist(), []))
        dic = dict(zip_iterator)

        wc = WordCloud(width=1920, height=1080, background_color='white', relative_scaling=0.8)
        wcloud = wc.generate_from_frequencies(dic)
        plt.imshow(wcloud)
        plt.axis("off")
        plt.show()
        plt.savefig("./results/%s" % self.q, format="png", dpi=1080)


if __name__ == '__main__':
    s = TfIdfSearch()
    s.search()
