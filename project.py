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

LANGUAGE = "en"
MAX_NGRAM_SIZE = 1 # Size of keywords, more than 1 to get phrases.
DEDUPLICATION_THRESHSOLD = 0.9 # Rate to avoid like-terms when picking out keywords. Should be less than 1.
NUM_OF_KEYWORDS = 3 # Number of keywords to retrieve per corpus.

imageObjectArray = []
topNodes = []
bottomNodes = []
edgesArray = []

class ImageObject:
    def __init__(self, corpus = [], keywords = []):
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
        extractor = KeywordExtractor(lan=LANGUAGE, n=MAX_NGRAM_SIZE, dedupLim=DEDUPLICATION_THRESHSOLD, top=NUM_OF_KEYWORDS, features=None)

        object.setKeywords(extractor.extract_keywords(object.getCorpusString()))

        print(object.getKeywords())

    setBottomNodes(getMasterKeywordList(imageObjectArray))
    
    i = 1
    for imageObject in imageObjectArray:
      keywords = imageObject.getKeywords()
      topNodes.append(i)
      for kw in keywords:
        if kw[0] in bottomNodes:
          edgesArray.append((i, kw[0]))
      i+=1
    
    B = nx.Graph()
    B.add_nodes_from(topNodes, bipartite=0)
    B.add_nodes_from(bottomNodes, bipartite=1)
    B.add_edges_from(edgesArray)
    left, right = nx.bipartite.sets(B, top_nodes = topNodes)
    pos = {}
    
    pos.update((node, (1, index)) for index, node in enumerate(left))
    pos.update((node, (2, index)) for index, node in enumerate(right))
    
    nx.draw(B, pos=pos, with_labels = True)
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
    

if __name__ == "__main__":
    main()