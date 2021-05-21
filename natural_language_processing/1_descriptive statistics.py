import sys

sys.path.append("./scripts")
from xml_corpus_reader import XMLCorpusReader

PATH_UNPROCESSED_CORPUS = "../corpus"
corpus = XMLCorpusReader(root=PATH_UNPROCESSED_CORPUS)
categories = corpus.categories()
categories.remove("presidential_speeches/obama")
categories.remove("presidential_speeches/trump")

categories_obama = [category for category in categories if "obama" in category]
categories_trump = [category for category in categories if "trump" in category]

print(corpus.describe(categories=categories_obama))  # 9'279'100 words
print(corpus.describe(categories=categories_trump))  # 8'313'658 words