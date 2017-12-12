#python example to infer document vectors from trained doc2vec model
import csv

import gensim.models as g
import codecs

#parameters

test_docs="/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/test_docs.txt"
output_file="/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
model = g.Doc2Vec.load("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/reuter-model.bin")


test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]

#infer test vectors
# output = open(output_file, "w")
# for d in test_docs:
#     output.write( " ".join([str(x) for x in model.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
# output.flush()
# output.close()

# f = open('output', 'w')
#
# sims = m.docvecs.most_similar([f])

vocab = list(model.wv.vocab)

word_vectors=(model[vocab])
# for w in word_vectors:
#     print (w)
print("num vectors:        ",len(word_vectors))

with open("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/doc2vec-master/test_data/test-body-vectors.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(word_vectors)

tokens = "CARLTON COMMUNICATIONS OPTIMISTIC FOR 1987|<Carlton Communications Plc> has started the current financial year well, with accounts for the first four months showing a healthy increase on the same period last year, and Chairman M.P. Green told the annual meeting he looked forward to 1987 with optimism. The issue of 4.6 mln shares in ADR form had now been successfully completed, he added. Carlton intended to increase its presence in the U.S. Which represented 50 pct of the world television market. Conditions worldwide in the television industry continued to look buoyant, the Chairman noted. REUTER".split()

new_vector = model.infer_vector(tokens)
sims = model.docvecs.most_similar([new_vector])


print(sims)
