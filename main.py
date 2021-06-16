from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tweets = pd.read_csv("./data/tweets.csv")
print(tweets.head(5))

tweet_keywords = tweets.keyword[:]
tweet_texts = tweets.text[:]

def show_tfidf(tfidf, vocab, filename):
    # [n_doc, n_vocab]
    plt.imshow(tfidf, cmap="YlGn", vmin=tfidf.min(), vmax=tfidf.max())
    plt.xticks(np.arange(tfidf.shape[1]), vocab, fontsize=6, rotation=90)
    plt.yticks(np.arange(tfidf.shape[0]), np.arange(1, tfidf.shape[0] + 1), fontsize=6)
    plt.tight_layout()
    plt.savefig("./results/%s.png" % filename, format="png", dpi=500)
    plt.show()


