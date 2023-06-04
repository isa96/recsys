import nltk
nltk.download('punkt')

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances

class ContentRecSys:
    def __init__(self, data, content_col, index_col=None):
        self.df = pd.read_csv(data, index_col=index_col)
        self.content_col = content_col
        self.encoder = None
        self.bank = None

    def fit(self):
        self.encoder = CountVectorizer(tokenizer=word_tokenize)
        self.bank = self.encoder.fit_transform(self.df[self.content_col])

    def recommend(self, idx, topk=10):
        content = self.df.loc[idx, self.content_col]
        code = self.encoder.transform([content])
        dist = cosine_distances(code, self.bank)
        rect_idx = dist.argsort()[0, 1:(topk+1)]
        return self.df.loc[rect_idx]
