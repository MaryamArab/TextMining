from __future__ import absolute_import, division, print_function
from gensim.models import word2vec
import codecs
import glob
import re
import nltk
from nltk.stem import WordNetLemmatizer
from stemming.porter2 import stem
from gensim.models.doc2vec import TaggedLineDocument, LabeledSentence, Doc2Vec
lemmatizer = WordNetLemmatizer()


documents = TaggedLineDocument('testWord2Vec.txt')

for s in documents:
    print (s)

regex = r"(?<!\d)[.,;:_-](?!\d)"
regexNum = r"(?<!\d)[0-9_.,!]*(?!\d)"
nltk.download('punkt')
f = open("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/My-Word2Vec/stopwords/english")
stopwords = f.read()
# articles = sorted(glob.glob("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/My-Word2Vec/data/Archive (1)/*.txt"))
articles = sorted(glob.glob("./*.txt"))

corpus_raw = u""
for article in articles:
    print("Reading '{0}' ... ".format(article))
    with codecs.open(article, "r", "utf-8") as book_file:
        corpus_raw += book_file.read()
    print("corpus is now {0} chars long".format(len(corpus_raw)))

print(len(corpus_raw))
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus_raw)

removedPunc_sentences = []
for r in raw_sentences:
    r = re.sub(regexNum, "", r, 0)
    r = re.sub(regex, "", r, 0)
    removedPunc_sentences.append(r)


def flatList(listoflists):
    return [item.lower() for list in listoflists for item in list]


def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]", " ", raw)
    words = clean.split()
    return words


sentences = []
for sentence in removedPunc_sentences:
    if len(sentence) > 0:
        sentences.append(sentence_to_wordlist(sentence.lower()))


print("*************start stem ************")
sentences = [[stem(word) for word in sent if word not in stopwords]
                         for sent in sentences]
se=[]
for s in sentences:
     print(s)
for s in sentences:
    se.append(' '.join(s))
print("done stemming")



sentences = flatList(sentences)
sentences = ' '.join(sentences)
for s in sentences:
    print(s)

with open('./lemmatize-out.txt', 'w') as file:
    for x in se:
        if x == 'reuter':
            file.write("\n")
        else:
            file.write(x)

#
documents = TaggedLineDocument('/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/lemmatize-out.txt')
model = Doc2Vec(documents, size=200, alpha=0.05, cbow_mean=1, workers=1, min_count=5, window=5)
print (model.most_similar("kodak ek continu drive market eastman kodak camera film product step boost imag diversifi inform system companirochest nyba compani unwrap seri electron data imag storag system creat vast comput librari document picturanalyst kodak start emerg market system sale expect reach billion dlrswarn pass system contribut bottom linekodak launch mln dlr adverti campaign promot imagimag market ibm data process market edgar greco vice presid manag kodak busi imag system divispct kodak billion dlrs sale photographi product kodak sale copier electron storag system busi product exceed billion dlrslaunch major attack market grecokodak perceiv busi bread butter wertheim analyst michael ellmanrichard schwarz follow kodak hutton compani profit margin slimmer sophist imag system consum photographi productcrit profitkodak announc commer avail inch optic diskkodak disk store equiv content file cabinet drawerreut audiotron corp ado offer debt audiotron corp regist secur exchang commiss offer mln dlrs convert"))
print (model)