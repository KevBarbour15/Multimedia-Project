import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from yake import KeywordExtractor
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import os
import PIL

LANGUAGE = "en"
MAX_NGRAM_SIZE = 1  # Size of keywords, more than 1 to get phrases.
# Rate to avoid like-terms when picking out keywords. Should be less than 1.
DEDUPLICATION_THRESHSOLD = 0.9
NUM_OF_KEYWORDS = 5  # Number of keywords to retrieve per corpus.

imageObjectArray = []
topNodes = []
bottomNodes = []
edgesArray = []

icons = {}


class ImageObject:
    def __init__(self, corpus=[], keywords=[], id=-1):
        self._corpus: list = corpus
        self._keywords: list = keywords
        self._id: int = id

    def getCorpus(self) -> list:
        return self._corpus

    def getCorpusString(self) -> str:
        """
        Return the corpus as a single string.
        """

        string = ""
        for word in self._corpus:
            string = string + " " + word

        return string

    def setKeywords(self, keywords) -> None:
        self._keywords = keywords

    def getKeywords(self) -> list:
        """
        Returns a list of tuples: (word, weight).
        """
        return self._keywords

    def setTFIDF(self, val) -> None:
        self._tfidf = val

    def getTFIDF(self) -> float:
        return self._tfidf

    def getImageID(self) -> int:
        return self._id


def main():
    responseList = []
    annotations = ["/annotations/Image_Annotations_Set_1.csv",
                   "/annotations/Image_Annotations_Set_2.csv",
                   "/annotations/Image_Annotations_Set_3.csv"]

    for file in annotations:
        df = pd.read_csv(os.getcwd() + file)
        rli = [df[str(col + 1)] for col in range(df.shape[1] - 1)]
        responseList.extend(rli)

    def preprocess_corpus(texts):
        englishStopWords = set(stopwords.words("english"))

        def removeStopsDigits(tokens):
            return [token.lower() for token in tokens if token not in englishStopWords and not token.isdigit() and token not in punctuation]

        return [removeStopsDigits(word_tokenize(text)) for text in texts]

    processedResponsesList = []

    for idx in range(len(responseList)):
        # print(preprocess_corpus(responseList[idx]))
        processedResponsesList.append(preprocess_corpus(responseList[idx]))

    lemmatizer = WordNetLemmatizer()
    lemmatziedResponsesList = []

    def posTagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    for response in processedResponsesList:

        lemmatizedWords = []
        for word in response:
            tagged = pos_tag(word)

            for tup in tagged:
                if posTagger(tup[1]) != None:
                    word = lemmatizer.lemmatize(tup[0], posTagger(tup[1]))
                    lemmatizedWords.append(word)

        lemmatziedResponsesList.append(lemmatizedWords)

    i = 1
    for lemmatziedResponse in lemmatziedResponsesList:
        imageObjectArray.append(ImageObject(lemmatziedResponse, [], i))
        i += 1

    for object in imageObjectArray:
        extractor = KeywordExtractor(
            lan=LANGUAGE, n=MAX_NGRAM_SIZE, dedupLim=DEDUPLICATION_THRESHSOLD, top=NUM_OF_KEYWORDS, features=None)

        object.setKeywords(extractor.extract_keywords(
            object.getCorpusString()))

    i = 0
    for response in lemmatziedResponsesList:
        wordCount = len(lemmatziedResponsesList[i])
        average = 0
        tf = 0
        idf = 0
        j = 1
        for kw in imageObjectArray[i].getKeywords():
            kwCount = 0
            for word in response:
                if kw[0] in word:
                    kwCount += 1

                tf = (kwCount / wordCount)
                idf = math.log(len(imageObjectArray) / 1)
            j += 1
            average += (tf * idf)

        average /= j
        imageObjectArray[i].setTFIDF(average)
        i += 1

    #create lists of the top 5 of each end of variance
    leastVariance = getLeast(imageObjectArray)
    mostVariance = getMost(imageObjectArray)

    # add the TFIDF score to lists for graph
    leastVarianceNum = []
    mostVarianceNum = []
    for image in leastVariance:
        leastVarianceNum.append(image.getTFIDF())
    for image in mostVariance:
        mostVarianceNum.append(image.getTFIDF())

    leastImageNum = []
    mostImageNum = []
    for image in leastVariance:
      leastImageNum.append("Image #{}".format(image.getImageID()))
    for image in mostVariance:
      mostImageNum.append("Image #{}".format(image.getImageID()))
  
    labels = list(zip(leastImageNum,mostImageNum))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    height = 0

    fig, ax = plt.subplots()
    least = ax.bar(x - width/2, leastVarianceNum,
                    width, label='Least Variance', color="black")
    most = ax.bar(x + width/2, mostVarianceNum, width,
                    label='Most Variance', color = "lightgrey")
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Image Rankings #5  ----->  #1')
    ax.set_ylabel('Average TF-IDF Score')
    ax.set_title(
        'Images with the Least Variance and Most Variance sorted by average TF-IDF of response')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(least, padding=3)
    ax.bar_label(most, padding=3)
    plt.show()


def getMasterKeywordList(objectArray: list):
    masterKeywordList = {}
    for imageObject in imageObjectArray:
        for kw in imageObject.getKeywords():
            masterKeywordList[kw[0]] = kw[1]

    return masterKeywordList


def getLeast(obarr):
    responseList = obarr
    final_list = []
    for i in range(0, 5):
        max1 = 0
        for j in range(len(responseList)):
            tfidf = float(responseList[j].getTFIDF())
            if max1 == 0:
                max1 = responseList[j]
            else:
                if tfidf > max1.getTFIDF():
                    max1 = responseList[j]

        responseList.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list


def getMost(obarr):
    responseList = obarr
    final_list = []
    for i in range(0, 5):
        max1 = 0
        for j in range(len(responseList)):
            tfidf = float(responseList[j].getTFIDF())
            if max1 == 0:
                max1 = responseList[j]
            else:
                if tfidf < max1.getTFIDF():
                    max1 = responseList[j]

        responseList.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list


def randomColor():
    rgb = []
    for i in range(3):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = [r, g, b]
    return rgb


def loadImages():
    directory = 'images'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            pathname, extension = os.path.splitext(f)
            fname = pathname.split('/')
            icons[fname[-1]] = f
            

if __name__ == "__main__":
    main()
