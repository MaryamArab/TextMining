from __future__ import absolute_import, division, print_function
import codecs
import glob
import re
import nltk
from itertools import chain, imap
import gensim.models.word2vec as w2v
import os

from gensim import corpora
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from stemming.porter2 import stem
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
# using regex to remove punctuations and numbers
regex = r"(?<!\d)[.,;:_-](?!\d)"
regexNum = r"(?<!\d)[0-9_.,!]*(?!\d)"
nltk.download('punkt')
f = open("stopwords/english")
stopwords = f.read()
articles=sorted(glob.glob("data_title/*.txt"))

corpus_raw =u""
for article in articles:
    print("Reading '{0}' ... ".format(article))
    with codecs.open(article,"r","utf-8") as book_file:
        corpus_raw += book_file.read()
    print ("corpus is now {0} chars long".format(len(corpus_raw)))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

removedPunc_sentences=[]
for r in raw_sentences:
    r = re.sub(regexNum, "", r, 0)
    r = re.sub(regex, "", r, 0)
    removedPunc_sentences.append(r)


for r in raw_sentences:
    print (r)


def flatList(listoflists):
        return [item for list in listoflists for item in list]

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


sentences = []
for sentence in removedPunc_sentences:
    if len(sentence) > 0:
        sentences.append(sentence_to_wordlist(sentence.lower()))
#for w in sentences:
#    print(w)

sentences = [[word for word in sent if word not in stopwords]
                         for sent in sentences]

sentences = flatList(sentences)
sentences = ' '.join(sentences)
print(sentences)
articles = glob.glob(os.path.join(os.getcwd(), "/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/Cleaner/CleanedData-seperated","*.txt"))
print(articles)
doc_set=[]
for file_path in articles:
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore')  as f_input:
        doc_set.append(f_input.read())
print("done writing to file")

# for d in thefile:
#     doc_set.append(d)
#
texts=[]
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    texts.append(tokens)

dictionary = corpora.Dictionary(texts)
corpus=[dictionary.doc2bow(text) for text in texts]
print("*******************************Training*********************************")

num_features = 300
min_word_count = 1
num_workers = 1
context_size = 7
downsampling = 1e-3
seed = 0
model = w2v.Word2Vec(
    sg=0,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
model.build_vocab(texts)

total_examples = sum([len(sentence) for sentence in sentences])
model.train(sentences, total_examples, epochs= model.iter)
print("Word2Vec vocabulary length:", total_examples)
model.save(os.path.join("trained","w2vModel.bin"))
model = w2v.Word2Vec.load(os.path.join("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/My-Word2Vec/trained","testModel.bin"))
model = w2v.Word2Vec.load(os.path.join("trained","w2vModel.bin"))
print("*****************************END TRAINING ******************************")

vocab = list(model.wv.vocab)
for v in vocab:
    print(v)
print(model.wv.most_similar('percent'))

# print(model.wv.most_similar(positive=['regan', 'worth'], negative=['digit']))
# ****************************************************
def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE( n_components=2, random_state=0)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    # plt.axis([-12, 0, 5, 20])
    plt.show()
# *****************************************
tsne_plot(model)
