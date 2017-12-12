from __future__ import absolute_import, division, print_function
import os
from sklearn.cluster import KMeans
from gensim.models import word2vec

import codecs
import glob
import re
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

regex = r"(?<!\d)[.,;:_-](?!\d)"
regexNum = r"(?<!\d)[0-9_.,!]*(?!\d)"
nltk.download('punkt')
f = open("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/AllData/stopwords/english")
stopwords = f.read()
articles=sorted(glob.glob("data/Archive (1)/*.txt"))

corpus_raw =u""
for article in articles:
    print("Reading '{0}' ... ".format(article))
    with codecs.open(article,"r","utf-8") as book_file:
        corpus_raw += book_file.read()
    print ("corpus is now {0} chars long".format(len(corpus_raw)))

print(len(corpus_raw))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)
# for r in raw_sentences:
#     print(r)
removedPunc_sentences=[]
for r in raw_sentences:
    r = re.sub(regexNum, "", r, 0)
    r = re.sub(regex, "", r, 0)
    removedPunc_sentences.append(r)
print("*************raw************")

print("*************raw************")


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
print("*************sentecne************")


print(len(sentences))
#
print("*************start stem ************")
sentences = [[lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='v') for word in sent if word not in stopwords]
                         for sent in sentences]
print("done stemming")
sentences = flatList(sentences)
sentences = ' '.join(sentences)

with open('data/Archive (1)/test-body_noStem-out.txt', 'w') as file:
    file.write(sentences)

for s in sentences:
    print(s)
print("*******************************Training*********************************")

raw_sentences= word2vec.Text8Corpus("data/Archive (1)/test-body_noStem-out.txt")

print("*******************************Training*********************************")

model = word2vec.Word2Vec(raw_sentences, size=200, alpha=0.05, cbow_mean=1, workers=1,min_count=5, window=5)
total_examples = sum([len(sentence) for sentence in raw_sentences])
model.train(raw_sentences, total_examples, epochs= model.iter)
print("Word2Vec vocabulary length:", total_examples)
model.save(os.path.join("trained","testModel-Final"))
model = word2vec.Word2Vec.load(os.path.join("trained","testModel-Final"))
test = 'earn'

print("Most similar words to \'"+test+"\' :")
print(model.most_similar(test, topn=20)[0][0])
print(model.most_similar(test, topn=20)[1][0])
print(model.most_similar(test, topn=20)[2][0])
print(model.most_similar(test, topn=20)[3][0])
print(model.most_similar(test, topn=20)[4][0])
print(model.most_similar(test, topn=20)[5][0])
print(model.most_similar(test, topn=20)[6][0])
print(model.most_similar(test, topn=20)[7][0])
print(model.most_similar(test, topn=20)[8][0])




vocab = list(model.wv.vocab)
# print (vocab)
# print(len(vocab))
word_vectors=(model[vocab])
# print(len(model[vocab]))
# print(len(word_vectors))
# *********************Save vectors to csv file********************

# with open("data/Archive (1)/test-body-vectors.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(word_vectors)


# ****************** Kmeans******************

def makeClusters(word_vectors,num_clusters):
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    for cluster in xrange(0, num_clusters):
        print("\nCluster %d" % cluster)

        # Find all of the words for that cluster number, and print them out
        words = []
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
                words.append(word_centroid_map.keys()[i])

        print(words)

# *************************************


# makeClusters(word_vectors, 51)


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




