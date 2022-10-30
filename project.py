import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from yake import KeywordExtractor
import os
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import PIL

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


def main():
    df = pd.read_csv(os.getcwd() + "/Image_Annotations.csv")

    responseList = [df[str(col + 1)] for col in range(df.shape[1] - 1)]

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
    
    setBottomNodes(getMasterKeywordList(imageObjectArray))
    
    #loads images into memory
    loadImages()
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}


    B = nx.Graph()
    B.add_nodes_from(bottomNodes, bipartite=1)

    #create edges
    i = 1
    for imageObject in imageObjectArray:
        keywords = imageObject.getKeywords()
        topNodes.append(i)
        title = str(i)
        B.add_node(i, image=images[title])
        for kw in keywords:
            if kw[0] in bottomNodes:
                B.add_edge(i, kw[0], image=images[title])
        i += 1
    
    #separate top and bottom nodes
    left, right = nx.bipartite.sets(B, top_nodes=bottomNodes)
    pos = {}
    
    i = 1
    for node in right:
        pos[node] = (2, i)
        i += 10

    i = 1
    for node in left:
        pos[node] = (1, i)
        i += 5
    
    fig, ax = plt.subplots()
    
    nx.draw_networkx_edges(
        B,
        pos=pos,
    )

    nx.draw_networkx_nodes(
        B,
        pos=pos,
        label=True
    )

    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.055
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
    for imageObject in imageObjectArray:
        for kw in imageObject.getKeywords():
            masterKeywordList[kw[0]] = kw[1]

    return masterKeywordList


def setBottomNodes(keywordsList):
    for kw in keywordsList:
        bottomNodes.append(kw)


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


if __name__ == "__main__":
    main()