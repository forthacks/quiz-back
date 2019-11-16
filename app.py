#!flask/bin/python
from flask import Flask, request
import numpy as np
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

wv = api.load('word2vec-google-news-300')
d = Dictionary(wv)
corpus = [d.doc2bow(line) for line in wv]
model = TfidfModel(corpus)

app = Flask(__name__)

@app.route('/')
def index():
    return "yeet"

@app.route('/sim')
def get_dist():
    a = request.args.get('a').split()
    b = request.args.get('b').split()
    
    # vectors
    vector_a = wv[a]
    vector_b = wv[b]

    # tfidf
    tfidf_a = model[d.token2id[a]]
    tfidf_b = model[d.token2id[b]]

    # dot product
    da = np.dot(vector_a, tfidf_a)
    db = np.dot(vector_b, tfidf_b)

    return np.inner(da, db)


if __name__ == '__main__':
    app.run(debug=True)