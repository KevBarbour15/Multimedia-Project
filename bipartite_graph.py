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
import random
import math
import os

LANGUAGE = "en"
MAX_NGRAM_SIZE = 1  # Size of keywords, more than 1 to get phrases.
# Rate to avoid like-terms when picking out keywords. Should be less than 1.
DEDUPLICATION_THRESHSOLD = 0.9
NUM_OF_KEYWORDS = 3  # Number of keywords to retrieve per corpus.

imageObjectArray = []
topNodes = []
bottomNodes = []
edgesArray = []

icons = {}


class ImageObject:
    def __init__(self, corpus=[], keywords=[]):
        self._corpus: list = corpus
        self._keywords: list = keywords

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

    for lemmatziedResponse in lemmatziedResponsesList:

        imageObjectArray.append(ImageObject(lemmatziedResponse))

    for object in imageObjectArray:
        extractor = KeywordExtractor(
            lan=LANGUAGE, n=MAX_NGRAM_SIZE, dedupLim=DEDUPLICATION_THRESHSOLD, top=NUM_OF_KEYWORDS, features=None)

        object.setKeywords(extractor.extract_keywords(
            object.getCorpusString()))

    B = nx.Graph()
    setBottomNodes(getMasterKeywordList(imageObjectArray))

    # create edges between each image object node and it's keywords
    i = 1
    for imageObject in imageObjectArray:
        keywords = imageObject.getKeywords()
        topNodes.append(i)
        color = randomColor()
        B.add_node(i)
        for kw in keywords:
            if kw[0] in bottomNodes:
                kwStrength = kw[1] * 10
                B.add_edge(i, kw[0], color=color, weight=kwStrength)
        i += 1

    left, right = nx.bipartite.sets(B, top_nodes=topNodes)
    pos = {}

    # Update position for node from each group
    i = 1
    for node in right:
        pos[node] = (2, i)
        i += 3.75

    i = (len(topNodes) * 7)
    for node in left:
        pos[node] = (1, i)
        i -= 7.5

    edges = B.edges()
    edgeColors = [B[u][v]['color'] for u, v in edges]
    edgeWeight = [B[u][v]['weight'] for u, v in edges]

    nx.draw(B, pos=pos, with_labels=True, node_color=(0.8, 0.8, 0.8),
            edge_color=edgeColors, width=edgeWeight)
    plt.show()


def getMasterKeywordList(objectArray: list):
    masterKeywordList = {}
    for imageObject in imageObjectArray:
        for kw in imageObject.getKeywords():
            masterKeywordList[kw[0]] = kw[1]

    return masterKeywordList


def setBottomNodes(keywordsList):
    for kw in keywordsList:
        bottomNodes.append(kw)


def randomColor():
    rgb = []
    for i in range(3):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = [r, g, b]
    return rgb


if __name__ == "__main__":
    main()