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
import PIL
import math
import random
import os

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
    def __init__(self, corpus=[], keywords=[]):
        self._corpus: list = corpus
        self._keywords: list = keywords
        self._tfidf: float = 0.00

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

    # loads images into memory
    loadImages()
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}

    # initialize the graph and set the bottom nodes of keywords
    B = nx.Graph()
    setBottomNodes(getMasterKeywordList(imageObjectArray))
    demoImages = getRandomImages()
    #demoImages = [1,10,19,39,67]
    
    # create edges
    i = 1
    for imageObject in imageObjectArray:
        keywords = imageObject.getKeywords()
        color = randomColor()
        if i in demoImages:
            topNodes.append(i)
            title = str(i)
            B.add_node(i, image=images[title])
            for kw in keywords:
                if kw[0] in bottomNodes:
                    kwStrength = kw[1] * 25
                    B.add_edge(i, kw[0], color=color, weight=kwStrength)
        i += 1

        # separate top and bottom nodes
    left, right = nx.bipartite.sets(B, top_nodes=topNodes)
    pos = {}

    i = 1
    for node in right:
        pos[node] = (2, i)
        i += 10.25

    i = (len(topNodes) * 40)
    for node in left:
        pos[node] = (1, i)
        i -= 45

    fig, ax = plt.subplots()

    edges = B.edges()
    edgeColors = [B[u][v]['color'] for u, v in edges]
    edgeWeight = [B[u][v]['weight'] for u, v in edges]
    nx.draw(B, pos=pos, with_labels=True, node_color=(0.8, 0.8, 0.8),
            edge_color=edgeColors, width=edgeWeight, font_size=15)

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform

    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.12
    icon_center = icon_size / 2.0

    isInt = True
    for n in B.nodes:
        try:
            int(n)
        except ValueError:
            isInt = False
        if isInt:
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))
            # get overlapped axes and plot icon
            a = plt.axes(
                [xa - icon_center, ya - icon_center, icon_size, icon_size])
            a.imshow(B.nodes[n]["image"])
            a.axis("off")

        isInt = True
    plt.show()


def getMasterKeywordList(objectArray: list):
    masterKeywordList = {}
    for imageObject in objectArray:
        for kw in imageObject.getKeywords():
            masterKeywordList[kw[0]] = kw[1]
    return masterKeywordList


def setBottomNodes(keywordsList):
    for kw in keywordsList:
        bottomNodes.append(kw)
        
def getRandomImages():
  randomImages = []
  i = 0
  while i < 5:
    randomImages.append(random.randint(1,70))
    i += 1
  print(randomImages)
  return randomImages


def loadImages():
    directory = 'images'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            pathname, extension = os.path.splitext(f)
            fname = pathname.split('/')
            icons[fname[-1]] = f


def parseName(filename):
    x = filename.split('/')
    print(x)

    return x


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