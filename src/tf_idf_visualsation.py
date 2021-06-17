from wordcloud import WordCloud

import matplotlib.pyplot as plt


class TfIdfVisualisation:
    def word_cloud_doc(self, doc_dict):
        # plot word cloud
        wc = WordCloud(width=1920, height=1080, background_color='white', relative_scaling=0.8,
                       min_font_size=1)
        wcloud = wc.generate_from_frequencies(doc_dict)
        plt.imshow(wcloud)
        plt.axis("off")
        plt.show()
        plt.savefig("./results/%s" % self.q, format="png", dpi=1080)
