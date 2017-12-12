from __future__ import absolute_import, division, print_function
import codecs
import glob
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

regex = r"(?<!\d)[.,;:_-](?!\d)"
regexNum = r"(?<!\d)[0-9_.,!]*(?!\d)"
nltk.download('punkt')
f = open("stopwords/english")
stopwords = f.read()
articles=articles=sorted(glob.glob("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/LDA/RawData/Reuter/ReuterTexts/*.txt"))

print(articles)

for article in articles:
    corpus_raw = u""
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
        if(r == 'Reuter'):
            removedPunc_sentences.append("\n")

    print("111111")
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

    print("22222")

    print(len(sentences))
    #
    print("*************start Lammatizing and stop word removal ************")
    sentences = [[lemmatizer.lemmatize(word, pos='v') for word in sent if word not in stopwords]
                             for sent in sentences]


    print("done Lammetizing and stop word removal")
    sentences = flatList(sentences)

    for s in sentences:
        print(s)
    sentences = ' '.join(sentences)
    with open('CleanData/20News-Cleaned.txt', 'w') as file:
        file.write(sentences)


#Replace vocabs
