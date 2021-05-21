from nltk.corpus.reader.api import CorpusReader, CategorizedCorpusReader
from nltk import sent_tokenize, wordpunct_tokenize, pos_tag, FreqDist
from bs4 import BeautifulSoup

import time
import codecs
import os
import xml.etree.ElementTree as ET


class XMLCorpusReader(CategorizedCorpusReader, CorpusReader):
    CAT_PATTERN = r".*(?=(\/(?<=\/)[^\/]*\.xml))"
    DOC_PATTERN = r"[\w ]+/[\w ]+/.+\.xml"

    def __init__(self, root, fileids=DOC_PATTERN, encoding="utf8", **kwargs):
        """
        Initialize the corpus reader. Categorization arguments (``cat_pattern``, ``cat_map``, and ``cat_file``)
        are passed to the ``CategorizedCorpusReader`` constructor. The remaining arguments are passed to the
        ``CorpusReader`` constructor.
        """

        # Add default category pattern if not passed into the class
        if not any(key.startswith("cat_") for key in kwargs.keys()):
            kwargs["cat_pattern"] = self.CAT_PATTERN

        # Initialize the NLTK corpus reader objects
        CorpusReader.__init__(self, root, fileids, encoding)
        CategorizedCorpusReader.__init__(self, kwargs)

    def resolve(self, fileids=None, categories=None):
        """
        Returns a list of fieleids or categories depending on what is passed to each internal corups reader
        function.
        """

        if fileids is not None and categories is not None:
            raise ValueError("Specify fileids or categories, not both")

        if categories is not None:
            return self.fileids(categories)
        return fileids

    def docs(self, fileids=None, categories=None):
        """
        Returns the complete text of an XML document, closing the document after done reading it and yielding
        it in a memory safe fashion.
        """

        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator loading one document into memory at a time
        for path, encoding in self.abspaths(fileids, include_encoding=True):
            with codecs.open(path, "r", encoding=encoding) as f:
                yield f.read()

    def sizes(self, fileids=None, categories=None):
        """
        Returns a list of tuples, the fileid and size on disk of the file.
        This function is used to detect oddly large files in the corpus.
        """

        # Resolve the fileids and the categories
        fileids = self.resolve(fileids, categories)

        # Create a generator, getting every path and computing filesize
        for path in self.abspaths(fileids):
            yield os.path.getsize(path)

    def paras(self, fileids=None, categories=None):
        """
        Parses the paragraphs from XML
        """

        for doc in self.docs(fileids, categories):
            soup = BeautifulSoup(doc, "lxml")
            for paragraph in soup.findAll("paragraph"):
                yield paragraph.text
            soup.decompose()

    def sents(self, fileids=None, categories=None):
        """
        Uses the built in sentence tokenizer of nltk to extract sentences from the paragraphs.
        """

        for paragraph in self.paras(fileids, categories):
            for sentence in sent_tokenize(paragraph):
                yield sentence

    def words(self, fileids=None, categories=None):
        """
        Uses the built in word tokenizer of nltk to extract tokens from the sentences.
        """

        for sentence in self.sents(fileids, categories):
            for token in wordpunct_tokenize(sentence):
                yield token

    def tokenize(self, fileids=None, categories=None):
        """
        Segments, tokenizes, and tags a document in the corpus
        """

        for paragraph in self.paras(fileids=fileids):
            yield [
                pos_tag(wordpunct_tokenize(sent)) for sent in sent_tokenize(paragraph)
            ]

    def describe(self, fileids=None, categories=None):
        """
        Performs a single pass of the corpus and returns a dictionary with a variety of metrics
        concernig the state of the corpus.
        """

        started = time.time()

        # Structures to perform counting.
        counts = FreqDist()
        tokens = FreqDist()

        # Perform single pass over paragraphs, tokenize and count
        for para in self.paras(fileids, categories):
            counts["paras"] += 1

            for sent in sent_tokenize(para):
                counts["sents"] += 1

                for word in wordpunct_tokenize(sent):
                    counts["words"] += 1
                    tokens[word] += 1

        # Compute the number of files and categories in the corpus
        n_fileids = len(self.resolve(fileids, categories) or self.fileids())
        n_topics = len(self.categories(self.resolve(fileids, categories)))

        # Return data structure with information
        return {
            "files": n_fileids,
            "topics": n_topics,
            "paras": counts["paras"],
            "sents": counts["sents"],
            "words": counts["words"],
            "vocab": len(tokens),
            "lexdiv": float(counts["words"]) / float(len(tokens)),
            "ppdoc": float(counts["paras"]) / float(n_fileids),
            "sppar": float(counts["sents"]) / float(counts["paras"]),
            "secs": time.time() - started,
        }