###########################################
## 1 - DESCRIPTIVE STISTICS
###########################################
"""
this script is used to get some basic statistics about the collecected data.
mainly the number of words of each category is important, as the difference
of the wordcount of each category between the presidency of obama and trump
has to be taken into account for later text analysis.
"""

import sys

sys.path.append("./helper_scripts")
from xml_corpus_reader import XMLCorpusReader

# initialize corpus reader
corpus = XMLCorpusReader(root="../corpus")

# show statistics about all categories speparated by presidency, except the presidential speeches
categories = corpus.categories()
categories.remove("presidential_speeches/obama")
categories.remove("presidential_speeches/trump")
categories_obama = [category for category in categories if "obama" in category]
categories_trump = [category for category in categories if "trump" in category]
print(corpus.describe(categories=categories_obama))
print(corpus.describe(categories=categories_trump))

# show statistics about each single category
for category in corpus.categories():
    print(corpus.describe(categories=category))