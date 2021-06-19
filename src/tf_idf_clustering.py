from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt

doc = pd.read_csv("../data/train.txt", sep=';').iloc[:, 0]
labels = pd.read_csv("../data/train.txt", sep=';').iloc[:, 1]
dictionary = {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4, 'surprise': 5}
label_index = [dictionary[x] for x in labels]

vectorizer = TfidfVectorizer(stop_words='english')
tf_idf = vectorizer.fit_transform(doc)

true_k = 6
random_state = 0

model = MiniBatchKMeans(n_clusters=true_k, random_state=random_state, max_iter=300)
model.fit(tf_idf)
# print(model.labels_)

# dimensionality reduction to 2D for visualisation
pca = PCA(n_components=2, random_state=random_state)
reduced_tf_idf = pca.fit_transform(tf_idf.toarray())

# reduce the cluster centres to 2D
reduced_cluster_centres = pca.transform(model.cluster_centers_)

# print(model.predict(tf_idf))
# plot
plt.scatter(reduced_tf_idf[:, 0], reduced_tf_idf[:, 1], c=model.predict(tf_idf))
# cluster centres
plt.scatter(reduced_cluster_centres[:, 0], reduced_cluster_centres[:, 1], marker='x', s=150, c='b')
plt.savefig("../results/%s" % "clustering", format="png", dpi=1080)
plt.show()



