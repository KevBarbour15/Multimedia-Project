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

image_object_array = []


class ImageObject:
    def __init__(self, id, response_set, response_synset, similarityScore):
        self._id: int = id
        self._similarityScore: float = similarityScore
        self.__response_set: dict = response_set
        self.__response_synset: dict = response_synset

    def get_image_id(self) -> int:
        return self._id

    def get_similarity_score(self) -> float:
        return self._similarityScore


def main():
    response_list = []
    annotations = ["/annotations/Image_Annotations_Set_1.csv",
                   "/annotations/Image_Annotations_Set_2.csv",
                   "/annotations/Image_Annotations_Set_3.csv"]

    for file in annotations:
        df = pd.read_csv(os.getcwd() + file)
        rli = [df[str(col + 1)] for col in range(df.shape[1] - 1)]
        response_list.extend(rli)

    def preprocess_corpus(texts):
        english_stop_words = set(stopwords.words("english"))

        def remove_stops_digits(tokens):
            return [token.lower() for token in tokens if token not in english_stop_words and not token.isdigit() and token not in punctuation]

        return [remove_stops_digits(word_tokenize(text)) for text in texts]

    processed_responses_list = []

    for idx in range(len(response_list)):
        processed_responses_list.append(preprocess_corpus(response_list[idx]))

    def pos_tagger(nltk_tag):
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

    def tagged_synset(word):
        wn_tag = pos_tagger(word[1])
        if wn_tag is None:
            return None
        try:
            return wordnet.synsets(word[0], wn_tag)[0]
        except:
            return None

    synset_response_list = []
    i = 1
    for response_set in processed_responses_list:
        response_synset = []
        for response in response_set:
            synset_response = []
            response = pos_tag(response)
            for word in response:
                synset_response.append(tagged_synset(word))
            response_synset.append(synset_response)
        synset_response_list.append(response_synset)

        similarityScore = set_response_similarity(
            i, response_set, response_synset)
        image_object_array.append(ImageObject(
            i, response_set, response_synset, similarityScore))
        i += 1

    least_variance = get_least(image_object_array)
    most_variance = get_most(image_object_array)

    least_variance_num = []
    most_variance_num = []
    for image in least_variance:
        least_variance_num.append(image.get_similarity_score())
    for image in most_variance:
        most_variance_num.append(image.get_similarity_score())

    # Create a list of the top 10 of image # for each end of variance with
    least_image_num = []
    most_image_num = []
    for image in least_variance:
        least_image_num.append("#{}".format(image.get_image_id()))
    for image in most_variance:
        most_image_num.append("#{}".format(image.get_image_id()))

    labels = list(zip(least_image_num, most_image_num))
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    least = ax.bar(x - width/2, least_variance_num,
                   width, label='Least Variance', color="lightgrey")
    most = ax.bar(x + width/2, most_variance_num, width,
                  label='Most Variance', color="black")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel(
        'Image Rankings from 10  ----->  1 \n (Each tuple represents 2 images, the first being least variance,'
        + ' the other being most variance)')
    ax.set_ylabel('Average Similarity Score')
    ax.set_title(
        'The Top 10 Images with the Least Variance and Most Variance')

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
    display_final_results(results_list, categories)

#----------------------------------------------------------------#


def display_final_results(results_list, categories):
    print("\n\n\n\n")
    print("***** Finals Results of General Categories *****")
    print()
    idx = 0
    for category in results_list:
        total = 0
        if idx == 9:
            print("\n***** Final Results of Specific Categories *****\n")

        for pic in category:
            total += image_object_array[pic - 1].get_similarity_score()
        total = total / len(category)
        print(
            "-- Category: {} -- Score: {}".format(categories[idx], round(total, 3)))
        print()
        idx += 1


def set_response_similarity(image_num, response_set, response_synset):
    score = 0.0
    count = 0
    # first check for matching words before checking for synonyms to catch words that do not have "synonyms"
    # this is to catch things like proper nouns that may not be in wordnet 
    for s1, s2 in itertools.combinations(response_set, 2):
        for s in s1:
            if s in s2:
                score += 1
                count += 1

    for s1, s2 in itertools.combinations(response_synset, 2):
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
    score = round(score, 5)
    print("IMAGE{}, SCORE {}".format(image_num, score))
    return score


def get_least(obarr):
    response_list = obarr.copy()
    final_list = []
    for i in range(0, 10):
        max1 = 0
        for j in range(len(response_list)):
            sim = float(response_list[j].get_similarity_score())
            if max1 == 0:
                max1 = response_list[j]
            else:
                if sim > max1.get_similarity_score():
                    max1 = response_list[j]

        response_list.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list


def get_most(obarr):
    response_list = obarr.copy()
    final_list = []
    for i in range(0, 10):
        max1 = 0
        for j in range(len(response_list)):
            sim = float(response_list[j].get_similarity_score())
            if max1 == 0:
                max1 = response_list[j]
            else:
                if sim < max1.get_similarity_score():
                    max1 = response_list[j]

        response_list.remove(max1)
        final_list.append(max1)

    final_list.reverse()
    return final_list


if __name__ == "__main__":
    main()
