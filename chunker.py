import os
import pandas as pd
from nltk.tokenize import PunktSentenceTokenizer
import nltk


def main():
    responseObjectList = []
    annotations = ["/annotations/Image_Annotations_Set_1.csv",
                   "/annotations/Image_Annotations_Set_2.csv",
                   "/annotations/Image_Annotations_Set_3.csv"]

    for file in annotations:
        df = pd.read_csv(os.getcwd() + file)
        rli = [df[str(col + 1)] for col in range(df.shape[1] - 1)]
        responseObjectList.extend(rli)

    for responseObject in responseObjectList:
        for response in responseObject:
            words = nltk.word_tokenize(response)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()



if __name__ == "__main__":
    main()