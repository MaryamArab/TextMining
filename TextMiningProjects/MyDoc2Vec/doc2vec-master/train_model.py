#python example to train doc2vec model (with or without pre-trained word embeddings)
import csv
import glob

import gensim.models as g
import logging
import numpy as np

#doc2vec parameters
import pandas as pd

#pretrained word embeddings
# pretrained_emb = "doc2vec-master/test_data/pretrained_word_embeddings.txt" #None if use without pretrained embeddings

#input corpus
path = '/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/Cleaner/Cleaned_NoStem/*.txt'
files = glob.glob(path)


testFile = '/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/Cleaner/test.txt'
taggedDocuments = '/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/taggedDocuments-ns-all-100.txt'
 # iterate over the list getting each file

test = ""
for fle in files:
    with open(fle) as f:
        test = test+(f.read())+'\n'




with open(testFile, "w") as text_file:
    text_file.write(test)
train_corpus = testFile

#output model
saved_path = "/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/reuter-model-ns-all-100.bin"

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




#train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)

for d in docs:
    print(d)

vector_size = 100
window_size = 7
min_count = 10
sampling_threshold = 1e-3
negative_size = 2
train_epoch = 20
dm = 0  # 0 = dbow; 1 = dmpv
worker_count = 1  # number of parallel processes

model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size,
                  dbow_words=1, dm_concat=1, iter=train_epoch)

#save model
model.save(saved_path)
model = g.Doc2Vec.load("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/reuter-model-ns-all-100.bin")

vocab = list(model.wv.vocab)

word_vectors=model[vocab]
# docvec = model.docvecs[3999]
# print(docvec)

my_df = pd.DataFrame(word_vectors)

my_df.to_csv('vectors-ns-all-100.csv', index=False, header=False)
# for v in word_vectors:
#     print(v.key)
