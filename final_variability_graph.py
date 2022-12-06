import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.corpus import wordnet
from nltk import pos_tag
import matplotlib.pyplot as plt
import numpy as np
import os
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

        similarityScore = setResponseSimilarity(i, responseSet, responseSynSet)
        imageObjectArray.append(ImageObject(
            i, responseSet, responseSynSet, similarityScore))
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
        'Top 10 Images with the Least Variance and Most Variance')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(least, padding=3)
    ax.bar_label(most, padding=3)
    plt.show()

    # lists containing the image numbers of images that correspond to the theme
    animal_pics = [1, 4, 6, 16, 18, 19, 21, 23, 39,
                   41, 44, 46, 47, 48, 49, 51, 55, 57, 58, 62, 69, 70]
    specific_location_pics = [2, 13, 21, 30, 34, 47, 51, 54, 61, 64, 68, 69]
    setting_pics = [5, 6, 17, 23, 36, 39, 46, 53, 56, 57, 58, 59, 60, 64]
    color_pics = [1, 2, 9, 18, 21, 42, 43, 46, 49, 54, 56, 59, 60, 69]
    people_pics = [3, 25, 26, 28, 29, 30, 33,
                   34, 38, 47, 55, 61, 62, 64, 66, 67]
    transportation_pics = [2, 3, 7, 16, 43, 50, 52, 54, 56, 62]
    action_pics = [1, 4, 5, 7, 8, 9, 10, 16,
                   19, 20, 25, 28, 29, 51, 52, 61, 66]
    artist_pics = [11, 12, 41, 48, 49, 62]
    tv_movie_pics = [15, 22, 24, 28, 31, 35, 37, 45, 50, 55, 67]
    food_pics = [9, 11, 13, 20, 37, 42, 45, 60, 64]
    weather_pics = [17, 36, 43, 44, 56, 58, 59]
    attire_pics = [8, 23, 24, 31, 32, 33, 34]
    legos_pics = [14, 26, 42, 68]
    blackcat_pics = [1, 23, 46, 69]
    pixar_pics = [15, 35, 37, 50]
    santa_pics = [3, 24, 28, 33, 34, 38, 61, 65]
    beetle_pics = [18, 43, 44, 56]
    jesus_pics = [25, 26, 27, 29, 30]

    results_list = [animal_pics, setting_pics, color_pics,
                    transportation_pics, action_pics, tv_movie_pics, food_pics, weather_pics, attire_pics,
                    legos_pics, blackcat_pics, pixar_pics, santa_pics, beetle_pics, artist_pics, people_pics, jesus_pics, specific_location_pics]

    categories = ["Animals", "Setting", "Colors",
                  "Transportation", "Actions / Activities", "Television / Movies", "Food", "Weather", "Attire", "Legos",
                  "Black Cats", "Pixar Movies", "Santa Claus", "Beetles", "Famous Artist Styles", "Famous People", "Jesus", "Notable Locations"]

    # display final results (average score of each category)
    displayFinalResults(results_list, categories)

#----------------------------------------------------------------#


def displayFinalResults(results_list, categories):
    print("\n\n\n\n")
    print("***** Finals Results of General Categories *****")
    print()
    idx = 0
    for category in results_list:
        total = 0
        if idx == 9:
            print("\n***** Final Results of Specific Categories *****\n")
            
        for pic in category:
            total += imageObjectArray[pic - 1].getSimilarityScore()
        total = total / len(category)
        print(
            "-- Category: {} -- Score: {}".format(categories[idx], round(total, 3)))
        print()
        idx += 1


def setResponseSimilarity(imageNum, responseSet, responseSynSet):
    score = 0.0
    count = 0
    # first check for matching words before checking for synonyms to catch names that do not have "synonyms"
    for s1, s2 in itertools.combinations(responseSet, 2):
        for s in s1:
            if s in s2:
                score += 1
                count += 1

    for s1, s2 in itertools.combinations(responseSynSet, 2):
        # filter out the kws without synsets:
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
    score = round(score,5)
    print("IMAGE{}, SCORE {}".format(imageNum, score))
    return score


def getLeast(obarr):
    responseList = obarr.copy()
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
    responseList = obarr.copy()
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
