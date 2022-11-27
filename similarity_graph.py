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
from nltk.stem import PorterStemmer
import numpy as np
import random
import math
import os
import PIL
import itertools

LANGUAGE = "en"
MAX_NGRAM_SIZE = 1  # Size of keywords, more than 1 to get phrases.
# Rate to avoid like-terms when picking out keywords. Should be less than 1.
DEDUPLICATION_THRESHSOLD = 0.9
NUM_OF_KEYWORDS = 6  # Number of keywords to retrieve per corpus.

imageObjectArray = []
topNodes = []
bottomNodes = []
edgesArray = []

icons = {}


class ImageObject:
    def __init__(self, id, responseSet, responseSynSet, similarityScore):
        self._id: int = id
        self._similarityScore: float = similarityScore
        self.__responseSet: dict = responseSet
        self.__responseSynSet: dict = responseSynSet

    def getImageID(self) -> int:
        return self._id

    def getSimilarityScore(self) -> float:
        return self._similarityScore

    def getResponseSet(self) -> dict:
        return self.__responseSet

    def getResponseSynSet(self) -> dict:
        return self.__responseSynSet


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
        processedResponsesList.append(preprocess_corpus(responseList[idx]))

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

    def taggedSynset(word):
        wn_tag = posTagger(word[1])
        if wn_tag is None:
            return None
        try:
            return wordnet.synsets(word[0], wn_tag)[0]
        except:
            return None

    synsetResponseList = []
    i = 1
    for responseSet in processedResponsesList:
        responseSynSet = []
        for response in responseSet:
            synsetResponse = []
            response = pos_tag(response)
            for word in response:
                synsetResponse.append(taggedSynset(word))
            responseSynSet.append(synsetResponse)
        synsetResponseList.append(responseSynSet)
        
        similarityScore = setResponseSimilarity(i,responseSet,responseSynSet)
        imageObjectArray.append(ImageObject(i, responseSet, responseSynSet, similarityScore))
        i += 1


    leastVariance = getLeast(imageObjectArray)
    mostVariance = getMost(imageObjectArray)
    
    # add the TFIDF score to lists for graph
    leastVarianceNum = []
    mostVarianceNum = []
    for image in leastVariance:
        leastVarianceNum.append(image.getSimilarityScore())
    for image in mostVariance:
        mostVarianceNum.append(image.getSimilarityScore())
    
    # create a list of the top 10 of image # for each end of variance with 
    leastImageNum = []
    mostImageNum = []
    for image in leastVariance:
        leastImageNum.append("#{}".format(image.getImageID()))
    for image in mostVariance:
        mostImageNum.append("#{}".format(image.getImageID()))

    labels = list(zip(leastImageNum, mostImageNum))
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    height = 0

    fig, ax = plt.subplots()
    least = ax.bar(x - width/2, leastVarianceNum,
                   width, label='Least Variance', color="lightgrey")
    most = ax.bar(x + width/2, mostVarianceNum, width,
                  label='Most Variance', color="black")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Image Rankings from 10  ----->  1')
    ax.set_ylabel('Average Similarity Score')
    ax.set_title(
        'Top 10 Images with the Least Variance and Most Variance sorted by average cosine similarity of response')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(least, padding=3)
    ax.bar_label(most, padding=3)
    plt.show()

    images_with_animals = [19, 45, 23]


def getAverageScore(picList):
  total = 0
  for image in picList:
    total += image.getSimilarityScore()
  
  return total

def setResponseSimilarity(count,responseSet, responseSynSet):
    score = 0.0
    count = 0
    # first check for matching words before checking for synonyms to catch names that do not have "synonyms"
    for s1, s2 in itertools.combinations(responseSet, 2):
        for s in s1:
            if s in s2:
                score += 1
                count += 1
                arb += 1

    for s1, s2 in itertools.combinations(responseSynSet, 2):
        # filter out the nones:
        synsets1 = [ss for ss in s1 if ss]
        synsets2 = [ss for ss in s2 if ss]

        for synset in synsets1:
            # Get the similarity value of the most similar word in the other sentence
            try:
                best_score = max([synset.path_similarity(ss)
                                 for ss in synsets2])
            except:
                best_score = None

            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1
    score = score / count
    score = round(score, 3)
    print("IMAGE{}, SCORE {}".format(count,score))
    return score

def getLeast(obarr):
    responseList = obarr
    final_list = []
    for i in range(0, 10):
        max1 = 0
        for j in range(len(responseList)):
            sim = float(responseList[j].getSimilarityScore())
            if max1 == 0:
                max1 = responseList[j]
            else:
                if sim > max1.getSimilarityScore():
                    max1 = responseList[j]

        responseList.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list


def getMost(obarr):
    responseList = obarr
    final_list = []
    for i in range(0, 10):
        max1 = 0
        for j in range(len(responseList)):
            sim = float(responseList[j].getSimilarityScore())
            if max1 == 0:
                max1 = responseList[j]
            else:
                if sim < max1.getSimilarityScore():
                    max1 = responseList[j]

        responseList.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list

if __name__ == "__main__":
    main()