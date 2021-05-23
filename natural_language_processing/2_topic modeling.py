###########################################
## 2 - TOPIC MODELING
###########################################
"""
this script is used for the topic analysis. multiple lda models are built
and tested. the best model is used for the topic analysis. the calculated
cv measures for the lda models can be found in the folder "data/topic_modeling"
"""

import sys
import os
import gensim
import gensim.corpora as corpora
import pandas as pd
import re
import spacy
import pickle

sys.path.append("./helper_scripts")

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from lda_tester import LDATester
from xml_corpus_reader import XMLCorpusReader


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    stop_words = stopwords.words("english")
    return [
        [word for word in simple_preprocess(str(doc)) if word not in stop_words]
        for doc in texts
    ]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
    return texts_out


def prepare_data(path_unprocessed_corpus, categories):
    print("STATUS: START PREPARING DATA")

    path_base = "data/topic_modeling/"
    path_id2word = path_base + "-".join(categories).replace("/", "_") + "-id2word.pckl"
    path_texts = path_base + "-".join(categories).replace("/", "_") + "-texts.pckl"
    path_corpus = path_base + "-".join(categories).replace("/", "_") + "-corpus.pckl"

    if (
        os.path.exists(path_id2word)
        and os.path.exists(path_texts)
        and os.path.exists(path_corpus)
    ):
        f = open(path_id2word, "rb")
        id2word = pickle.load(f)
        f.close()
        f = open(path_texts, "rb")
        texts = pickle.load(f)
        f.close()
        f = open(path_corpus, "rb")
        corpus = pickle.load(f)
        f.close()

        print("STATUS: FINISHED PREPARING DATA")
        return (id2word, texts, corpus)

    corpus = XMLCorpusReader(root=path_unprocessed_corpus)

    docs = pd.DataFrame(columns=["id", "text"])
    for fileid in corpus.fileids(categories=categories):
        doc = {
            "id": fileid,
            "text": " ".join(list(corpus.paras(fileids=fileid))),
        }
        docs = docs.append(doc, ignore_index=True)

    # remove punctuation
    docs["text_processed"] = docs["text"].map(lambda x: re.sub("[,\.!?]", "", x))
    # convert text to lowercase
    docs["text_processed"] = docs["text_processed"].map(lambda x: x.lower())

    data = docs.text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    # build bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # form bigrams
    data_words_bigrams = make_bigrams(texts=data_words_nostops, bigram_mod=bigram_mod)
    # lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(
        # data_words_bigrams, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]
        data_words_bigrams,
        allowed_postags=["NOUN"],
    )

    # create dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # create corpus
    texts = data_lemmatized
    # term document frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    f = open(path_id2word, "wb")
    pickle.dump(id2word, f)
    f.close()
    f = open(path_texts, "wb")
    pickle.dump(texts, f)
    f.close()
    f = open(path_corpus, "wb")
    pickle.dump(corpus, f)
    f.close()

    print("STATUS: FINISHED PREPARING DATA")
    return (id2word, texts, corpus)


if __name__ == "__main__":
    PATH_UNPROCESSED_CORPUS = "../corpus"
    PATH_PROCESSED_CORPUS = "../pickled_corpus"

    """
    PRESIDENTIAL SPEECHES --> OBAMA
    """
    # prepare data
    categories = ["presidential_speeches/obama"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)

    # build final model
    # relevance metric: 0.17
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=41,
        alpha=0.01,
        beta=0.31,
    )

    """
    PRESIDENTIAL SPEECHES --> TRUMP
    """
    # prepare data
    """
    categories = ["presidential_speeches/trump"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.1
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=41,
        alpha=0.01,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF ENERGY --> OBAMA --> NEWS, BLOG POSTS, MEDIA ADVISORIES
    """
    # prepare data
    """
    categories = [
        "department_of_energy/obama/news",
        "department_of_energy/obama/blog_posts",
        "department_of_energy/obama/media_advisories",
    ]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.17
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=101,
        alpha=0.61,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF ENERGY --> TRUMP --> NEWS, BLOG POSTS, MEDIA ADVISORIES
    """
    # prepare data
    """
    categories = [
        "department_of_energy/trump/news",
        "department_of_energy/trump/blog_posts",
        "department_of_energy/trump/media_advisories",
    ]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.17
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=401,
        alpha=0.61,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF ENERGY --> OBAMA --> SPEECHES
    """
    # prepare data
    """
    categories = ["department_of_energy/obama/speeches"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.62
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=21,
        alpha=0.01,
        beta=0.61,
    )
    """

    """
    DEPARTMENT OF ENERGY --> TRUMP --> SPEECHES
    """
    # prepare data
    """
    categories = ["department_of_energy/trump/speeches"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.24
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=51,
        alpha=0.91,
        beta=0.31,
    )
    """

    """
    DEPARTMENT OF HOMELAND SECURITY --> OBAMA --> PRESS RELEASES, BLOG POSTS, MEDIA ADVISORIES
    """
    # prepare data
    """
    categories = [
        "department_of_homeland_security/obama/press_releases",
        "department_of_homeland_security/obama/blog_posts",
        "department_of_homeland_security/obama/media_advisories",
    ]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.38
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=81,
        alpha=0.01,
        beta=0.61,
    )
    """

    """
    DEPARTMENT OF HOMELAND SECURITY --> TRUMP --> PRESS RELEASES, BLOG POSTS, MEDIA ADVISORIES
    """
    # prepare data
    """
    categories = [
        "department_of_homeland_security/trump/press_releases",
        "department_of_homeland_security/trump/blog_posts",
        "department_of_homeland_security/trump/media_advisories",
    ]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.5
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=51,
        alpha=0.31,
        beta=0.31,
    )
    """

    """
    DEPARTMENT OF HOMELAND SECURITY --> OBAMA --> SPEECHES
    """
    # prepare data
    """
    categories = ["department_of_homeland_security/obama/speeches"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.6
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=13,
        alpha=0.31,
        beta=0.31,
    )
    """

    """
    DEPARTMENT OF HOMELAND SECURITY --> TRUMP --> SPEECHES
    """
    # prepare data
    """
    categories = ["department_of_homeland_security/trump/speeches"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.53
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=21,
        alpha=0.01,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF STATE --> OBAMA --> PRESS RELEASES
    """
    # prepare data
    """
    categories = ["department_of_state/obama/press_releases"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.33
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=51,
        alpha=0.01,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF STATE --> TRUMP --> PRESS RELEASES
    """
    # prepare data
    """
    categories = ["department_of_state/trump/press_releases"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.16
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=51,
        alpha=0.01,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF STATE --> OBAMA --> REMARKS_KERRY
    """
    # prepare data
    """
    categories = ["department_of_state/obama/remarks_kerry"]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.2
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=41,
        alpha=0.31,
        beta=0.01,
    )
    """

    """
    DEPARTMENT OF STATE --> TRUMP --> SPEECHES POMPEO, REMARKS TILLERSON
    """
    # prepare data
    """
    categories = [
        "department_of_state/trump/speeches_pompeo",
        "department_of_state/trump/remarks_tillerson",
    ]
    id2word, texts, corpus = prepare_data(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
    )
    lda_tester = LDATester(corpus=corpus, texts=texts, id2word=id2word)
    """

    # build final model
    # relevance metric: 0.2
    """
    lda_tester.compute_lda(
        file_name="-".join(categories).replace("/", "_"),
        n_topics=31,
        alpha=0.31,
        beta=0.01,
    )
    """

    # optimize number of topics of an lda model
    """
    lda_tester.opt_lda(
        file_name="-".join(categories).replace("/", "_") + "-NTOPCIS",
        min_topics=1,
        max_topics=1001,
        step_size=100,
        alpha=[0.01],
        beta=[0.1],
    )
    """

    # optimize hyperparameters of an lda model
    """
    lda_tester.opt_lda(
        file_name="-".join(categories).replace("/", "_") + "-HYPERPARAMETERS",
        min_topics=21,
        max_topics=21,
        step_size=1,
        alpha=list(np.arange(0.01, 1, 0.3)),
        beta=list(np.arange(0.01, 1, 0.3)),
    )
    """