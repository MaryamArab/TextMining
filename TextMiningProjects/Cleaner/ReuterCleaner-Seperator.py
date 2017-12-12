from __future__ import absolute_import, division, print_function


import codecs
import glob
import re
import nltk


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# ff= "21"
regex = r"(?<!\d)[.,;:_-](?!\d)"
regexNum = r"(?<!\d)[0-9_.,!]*(?!\d)"
nltk.download('punkt')
f = open("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/AllData/stopwords/english")
stopwords = f.read()
# articles=sorted(glob.glob("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/DocToVec/MyDoc2Vec/Cleaner/DataToClean/Output/reut0"+ff+"/*.txt"))
i=8868
for indx in range(10, 22):
    ff = str(indx)
    articles = sorted(glob.glob("/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/TextMiningProjects/MyDoc2Vec/Cleaner/DataToClean/Output/reut0" + ff + "/*.txt"))
    for article in articles:
        i=i+1
        corpus_raw = u""
        print("Reading '{0}' ... ".format(article))
        with codecs.open(article, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
        # print("corpus is now {0} chars long".format(len(corpus_raw)))
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(corpus_raw)
        removedPunc_sentences = []
        for r in raw_sentences:
            r = re.sub(regexNum, "", r, 0)
            r = re.sub(regex, "", r, 0)
            removedPunc_sentences.append(r)

        def flatList(listoflists):
            return [item for list in listoflists for item in list]


        def sentence_to_wordlist(raw):
            clean = re.sub("[^a-zA-Z]", " ", raw)
            words = clean.split()
            return words


        sentences = []
        for sentence in removedPunc_sentences:
            if len(sentence) > 0:
                sentences.append(sentence_to_wordlist(sentence.lower()))



        sentences = [[lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='v') for word in sent if word not in stopwords]
                     for sent in sentences]


        sentences = flatList(sentences)

        sentences = ' '.join(sentence
                             .replace('reuter', '\n')
                             .replace('mln', 'million')
                             .replace('dlrs', 'dollar')
                             .replace('shrs', 'share')
                             .replace('shr', 'share')
                             .replace('mthly', 'monthly')
                             .replace('mths', 'months')
                             .replace('avg', 'average')
                             .replace('qtly', 'quarterly')
                             for sentence in sentences)
        with open('/Users/maryam/Documents/Courses/Fall 2017/Data Mining/Project/Python/Cleaner/CleanedData-seperated/'+ff+'-' +str(i)+'.txt', 'w') as file:
            file.write(sentences)



print(i)

