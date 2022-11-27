
from nltk import word_tokenize, pos_tag, ne_chunk, sent_tokenize

def main():

    sentences_array = ["A giant squad attacking New York City.", "A LEGO figure of Batman.", "Michael Jackson at McDonalds"]

    for sentence in sentences_array:
        for sent in sent_tokenize(sentence):
            for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
                if hasattr(chunk, 'label'):
                    print(chunk.label(), ' '.join(c[0] for c in chunk))


if __name__ == "__main__":
    main()