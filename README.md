# TF_IDF_search_engine
## src/tf_idf_search.py:
This search engine only use TF-IDF measure with provided documents, query and desired range of results. A default document, data/train.txt, is fed if no assigned document.
### Result:
```python
s = TfIdfSearch(doc=(pd.read_csv("../data/train.txt", sep=';').iloc[:, 0]), q="i am so excited to see it")
s.search(0, 3)
```
will return:
```python
['i feel so excited about it', 'i think i was feeling so excited today', 'i feel so excited for college']
```

## src/tf_idf_clustering.py:
Using Batch K-means algorithm to visualise the spread of data, data/train.txt.
### Result:
![](https://github.com/serfinxx/TF_IDF_search_engine/blob/master/results/clustering.png)
### Conclusion:
It can be seen the clusters are overlapping to each others. It can be inferred that this model's performance could be better if using different parameters. Other ways of classification with deep learning will be added in the future.
# Acknowledgements:
Elvis - https://lnkd.in/eXJ8QVB & Hugging face team with [License CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
