import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from yake import KeywordExtractor
from collections import Counter
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, corpus=[], keywords={}, id=-1, keywordstfidf = {}):
        self._corpus: list = corpus
        self._keywords: dict = keywords
        self._id: int = id
        self._keywordstfidf: list = keywordstfidf

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

    def addKeyword(self, key, val) -> None:
        self._keywords[key] = val

    def getKeywords(self) -> dict:
        return self._keywords

    def addKeywordTfidf(self,key, val) -> None:
        self._keywordstfidf[key] = val
        
    def getKeywordsTfidf(self) -> dict:
      return self._keywordstfidf

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
        #print(preprocess_corpus(responseList[idx]))
        processedResponsesList.append(preprocess_corpus(responseList[idx]))

    lemmatizer = WordNetLemmatizer()
    lemmatizedResponsesList = []

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
    
    responsesWithTfidf = []
    for responses in processedResponsesList:
        lemmatizedWords = []
        responseSetAfterLem = []
        
        for response in responses:
            tagged = pos_tag(response)
            responseAfterLem = []
  
            for tup in tagged:
    
                if posTagger(tup[1]) != None:
                    word = lemmatizer.lemmatize(tup[0], posTagger(tup[1]))
                    responseAfterLem.append(word)
                    lemmatizedWords.append(word)
            
            responseSetAfterLem.append(responseAfterLem)
  
        lemmatizedResponsesList.append(lemmatizedWords)
        DF = getDF(responseSetAfterLem)
        TFIDF = getTFIDF(responseSetAfterLem, DF)
        responsesWithTfidf.append(TFIDF)
        
        
    i = 1
    for lemmatizedResponse in lemmatizedResponsesList:
        imageObjectArray.append(ImageObject(lemmatizedResponse, {}, i, responsesWithTfidf[i-1]))
        i += 1

    for object in imageObjectArray:
        extractor = KeywordExtractor(
            lan=LANGUAGE, n=MAX_NGRAM_SIZE, dedupLim=DEDUPLICATION_THRESHSOLD, top=NUM_OF_KEYWORDS, features=None)

        for kw in extractor.extract_keywords(object.getCorpusString()):
            object.getKeywords()[kw[0]] = kw[1]

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
                if kw in bottomNodes:
                    kwWeight = ((1-imageObject.getKeywordsTfidf()[kw]) * 5)
                    B.add_edge(i, kw, color=color, weight=kwWeight)
        i += 1
        # separate top and bottom nodes
    left, right = nx.bipartite.sets(B, top_nodes=topNodes)
    pos = {}

    # create node postions for each vertex
    i = (len(right)*9.5)
    for node in right:
        pos[node] = (2, i)
        i -= 9.5

    i = (len(topNodes) * 40)
    for node in left:
        pos[node] = (1, i)
        i -= 45

    fig, ax = plt.subplots()

    edges = B.edges()
    edgeColors = [B[u][v]['color'] for u, v in edges]
    edgeWeight = [B[u][v]['weight'] for u, v in edges]
    nx.draw(B, pos=pos, with_labels=True, node_color=(0.8, 0.8, 0.8),
            edge_color=edgeColors, font_size=15,width = edgeWeight)

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
    masterKeywordList = []
    for imageObject in objectArray:
        for kw in imageObject.getKeywords():
            if kw not in masterKeywordList:
                masterKeywordList.append(kw)
    return masterKeywordList


def setBottomNodes(keywordsList):
    for kw in keywordsList:
        bottomNodes.append(kw)

# returns a dict of the keywords and their respective count in the response set
def getDF(responsesList):
  DF = {}
  for i in range(len(responsesList)):
    tokens = responsesList[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
  for i in DF:
      DF[i] = len(DF[i])
  return DF

def getTFIDF(responsesList, DF):
  tf_idf = {}
  
  for response in responsesList:
    tokens = response
    counter = Counter(tokens + response)
    for token in np.unique(tokens):
        #print("Word: {}".format(token))
        tf = counter[token]/len(response)
        #print("TF: {}".format(tf))
        df = DF[token]
        #print("DF: {}".format(df))
        idf = np.log(len(responsesList)/(df))
        #print("IDF: {}".format(idf))
        tf_idf[token] = tf*idf
        #print("TF_IDF: {}".format(tf_idf[token]))
  
  return tf_idf
  

def getRandomImages():
    randomImages = []
    i = 0
    while i < 5:
        randomImages.append(random.randint(1, 70))
        i += 1
    #print(randomImages)
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
