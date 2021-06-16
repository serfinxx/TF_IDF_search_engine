from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

# {'anger': 0, 'love': 1, 'fear': 2, 'joy': 3, 'sadness': 4,'surprise': 5}
class TfIdfClustering:
    def plot(self):
        # create k-means model to plot
        num_clusters = 6
        max_iterations = 300
        labels_color_map = {
            0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
            5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
        }
        pca_num_components = 2
        tsne_num_components = 2

        clustering_model = KMeans(
            n_clusters=num_clusters,
            max_iter=max_iterations,
            precompute_distances="auto",
            n_jobs=-1
        )

        labels = clustering_model.fit_predict(self.tf_idf)
        dense_tfidf = self.tf_idf.todense()

        reduced_data = PCA(n_components=pca_num_components).fit_transform(dense_tfidf)

        fig, ax = plt.subplots()
        for index, instance in enumerate(reduced_data):
            # print instance, index, labels[index]
            pca_comp_1, pca_comp_2 = reduced_data[index]
            color = labels_color_map[labels[index]]
            ax.scatter(pca_comp_1, pca_comp_2, c=color)
        plt.show()

        # t-SNE plot
        embeddings = TSNE(n_components=tsne_num_components)
        Y = embeddings.fit_transform(dense_tfidf)
        plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
        plt.show()
