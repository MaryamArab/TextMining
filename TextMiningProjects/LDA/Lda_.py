import codecs
import glob

import os

import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import RegexpTokenizer
import gensim
from gensim import corpora



tokenizer = RegexpTokenizer(r'\w+')

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))

print("********************************")

documents = dataset.data
# thefile = open('/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/LDA/CleanData/20News-Cleaned.txt', 'r')
# articles = sorted(glob.glob("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/Cleaner/CleanData/*.txt"))
# thefile = open('/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/LDA/CleanData/20News-Cleaned.txt', 'r')
# for item in documents:
#   thefile.write("%s\n" % item.encode('ascii', 'ignore').decode('ascii'))
articles = glob.glob(os.path.join(os.getcwd(), "/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/Cleaner/CleanedData-seperated","*.txt"))
print(articles)
doc_set=[]
for file_path in articles:
    with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as f_input:
        doc_set.append(f_input.read())
print("done writing to file")
#
# for d in thefile:
#     doc_set.append(d)

texts=[]
for i in doc_set:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    texts.append(tokens)
print(texts)
dictionary = corpora.Dictionary(texts)
corpus=[dictionary.doc2bow(text) for text in texts]

# print(texts[0])

# no_features = 1000

# # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tf = tf_vectorizer.fit_transform(documents)
# tf_feature_names = tf_vectorizer.get_feature_names()
#
no_topics = 20

ldamodel= gensim.models.LdaModel(corpus,num_topics=14, id2word=dictionary,passes=30,minimum_probability=0.4)
Topics= ldamodel.show_topics()

# # Run LDA
# lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
# lda = gensim.models.ldamodel.LdaModel()
no_top_words = 10

# display_topics(lda, tf_feature_names, no_top_words)

print(ldamodel.print_topics(num_topics=14, num_words=10))
print(ldamodel.get_document_topics(corpus[0]))
print("************** LDA Done !!! ****************")
ldamodel.save(os.path.join("trained","ldaModel"))
# model = word2vec.Word2Vec.load(os.path.join("trained","ldaModel"))
print("*********making matrix*************")
mixture =[dict(ldamodel[x]) for x in corpus]
pd.DataFrame(mixture).to_csv("doc_topic-0.4.csv")
print("************** Matrix Done !!! ****************")


# no_features = 200
# #
# # NMF is able to use tf-idf
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(corpus)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#
# no_topics = 14
#
# # Run NMF
# nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# no_top_words = 10
# display_topics(nmf, tfidf_feature_names, no_top_words)

print("************************* NMF DOne !!! REALLY???? :O :O :O  *****************")


#
