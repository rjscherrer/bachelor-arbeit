import sys

sys.path.append("./scripts")
from xml_corpus_reader import XMLCorpusReader
from preprocessor import Preprocessor

if __name__ == "__main__":
    # settings
    PATH_UNPROCESSED_CORPUS = "../corpus"
    PATH_PROCESSED_CORPUS = "../pickled_corpus"

    # open corpus and map files to categories
    corpus = XMLCorpusReader(root=PATH_UNPROCESSED_CORPUS)
    # print categories to check if they are correct
    print(corpus.categories())

    # perform preprocessing:
    # text is devided into single words and tagged
    # lemmatizing and such things is done in a later step
    print("STATUS: Preprocessing in progress...")
    preprocessor = Preprocessor(corpus=corpus, target=PATH_PROCESSED_CORPUS)
    preprocessor.transform()
    print("STATUS: Preprocessing done")