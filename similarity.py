from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    print(word)
    print(tag)
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        print(wn.synsets(word, wn_tag)[0])
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    print(sentence1)
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    print(sentence1)
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    print(synsets1)
    print(synsets2)
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = max([synset.path_similarity(ss) for ss in synsets2])
        
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score


sentences = [
    "Dirtbike rider jumping.",
    "Dirt bike rider doing a jump over a hill",
    
    "Black cat catching a bird"
]

focus_sentence = "A black cat catching a bird"


def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    return (sentence_similarity(sentence1, sentence2) + sentence_similarity(sentence2, sentence1)) / 2


for sentence in sentences:
    print("SymmetricSimilarity(\"%s\", \"%s\") = %s" % (focus_sentence,
          sentence, symmetric_sentence_similarity(focus_sentence, sentence)))
    print("SymmetricSimilarity(\"%s\", \"%s\") = %s" % (
        sentence, focus_sentence, symmetric_sentence_similarity(sentence, focus_sentence)))
    print()
