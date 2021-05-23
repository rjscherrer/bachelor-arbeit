###########################################
## 3 - RHETORICAL ANALYSIS
###########################################
"""
this script is used for the rhetorical analysis. the analysis consists of the follwoing parts:

- analysis 1: rhetorical analysis with all presidential phrases
- analysis 2: evaluating phrases which could have a bad impact on the result
- analysis 3: rhetorical analysis with removed phrases which could have a bad impact on the result (run together with analysis 1)

the start of each part is indicated by a comment in the form "START ANALYSIS X" while the end of each part is indicated
by a comment in the form "END ANALYSIS X". as they depend on each other they were not separated in order to prevent redundant code. 
initially analysis 2 and analysis 3 are commented out. to run them they have to be uncommented.

the processed data can be found in the folder "data/rhetorical_analysis"
"""

import sys
import os
import gensim
import pandas as pd
import re
import spacy
import pickle
import tqdm

sys.path.append("./helper_scripts")

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from xml_corpus_reader import XMLCorpusReader
from tqdm import tqdm


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
    # initialize spacy 'en' model, keeping only tagger component
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    pbar = tqdm(total=len(texts))
    texts_out = []
    for sent in texts:
        doc = nlp(sent)
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )
        pbar.update(1)
    pbar.close()

    return texts_out


def tag_phrase(phrarse):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    return nlp(phrarse.decode("UTF-8"))[0].pos_


def build_ngrams(path_unprocessed_corpus, categories, bigram_vocab, trigram_vocab):
    print("STATUS: START BUILDING NGRAMS")
    path_base = "data/rhetorical_analysis/"
    path_bigrams = path_base + "-".join(categories).replace("/", "_") + "-bigrams.pckl"
    path_trigrams = (
        path_base + "-".join(categories).replace("/", "_") + "-trigrams.pckl"
    )

    if os.path.exists(path_bigrams) and os.path.exists(path_trigrams):
        f = open(path_bigrams, "rb")
        bigram_dict = pickle.load(f)
        f.close()
        f = open(path_trigrams, "rb")
        trigram_dict = pickle.load(f)
        f.close()

        print("STATUS: FINISHED BUILDING NGRAMS (LOADING EXISTING DATA)")
        return (bigram_dict, trigram_dict)

    corpus = XMLCorpusReader(root=path_unprocessed_corpus)

    docs = pd.DataFrame(columns=["id", "text"])
    for fileid in corpus.fileids(categories=categories):
        doc = {
            "id": fileid,
            "text": " ".join(list(corpus.paras(fileids=fileid))),
        }
        docs = docs.append(doc, ignore_index=True)

    # remove punctuation
    print("STATUS: REMOVE PUNCTUATION")
    docs["text_processed"] = docs["text"].map(lambda x: re.sub("[,\.!?]", "", x))

    # convert text to lowercase
    print("STATUS: CONVERT TO LOWERCASE")
    docs["text_processed"] = docs["text_processed"].map(lambda x: x.lower())

    # lemmatize
    print("STATUS: LEMMATIZE")
    docs["text_processed"] = lemmatization(
        texts=docs["text_processed"],
        allowed_postags=["NOUN", "ADJ", "VERB", "ADV"],
    )

    # perpare data for ngram
    data = docs.text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    # remove Stop Words
    print("STATUS: REMOVE STOPWORDS")
    data_words_nostops = remove_stopwords(data_words)

    # build bigrams
    print("STATUS: START BUILDING BIGRAMS")
    phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    relevant_bigrams = gensim.models.phrases.Phraser(phrases).phrasegrams
    bigram_dict = {
        b"_".join(k): phrases.vocab[b"_".join(k)]
        for k, v in relevant_bigrams.items()
        if phrases.vocab[b"_".join(k)] < 1000
        and bigram_vocab.get(b"_".join(k), 0) < 7300
    }

    # build trigrams
    print("STATUS: START BUILDING TRIGRAMS")
    phrases = gensim.models.Phrases(
        phrases[data_words_nostops], min_count=1, threshold=10
    )
    relevant_trigrams = gensim.models.phrases.Phraser(phrases).phrasegrams
    trigram_dict = {
        b"_".join(k): phrases.vocab[b"_".join(k)]
        for k, v in relevant_trigrams.items()
        if b"_".join(k).count(b"_") == 2
        and phrases.vocab[b"_".join(k)] < 1000
        and trigram_vocab.get(b"_".join(k), 0) < 480
    }

    f = open(path_bigrams, "wb")
    pickle.dump(bigram_dict, f)
    f.close()
    f = open(path_trigrams, "wb")
    pickle.dump(trigram_dict, f)
    f.close()

    print("STATUS: FINISHED BUILDING NGRAMS")
    return (bigram_dict, trigram_dict)


def calc_phrase_statistics(phrases_obama, phrases_trump, phrases_corpus):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    distinct_phrases = list(
        set(list(phrases_obama.keys()) + list(phrases_trump.keys()))
    )

    phrase_occ_obama = sum(phrases_obama.values())
    phrase_occ_trump = sum(phrases_trump.values())

    phrase_statistics = {
        phrase: (
            (
                abs(
                    (
                        phrases_trump.get(phrase, 0)
                        * (phrase_occ_obama - phrases_obama.get(phrase, 0))
                    )
                    - (
                        phrases_obama.get(phrase, 0)
                        * (phrase_occ_trump - phrases_trump.get(phrase, 0))
                    )
                )
                ** 2.5
            )
            / (
                (phrases_trump.get(phrase, 0) + phrases_obama.get(phrase, 0))
                * (
                    phrases_trump.get(phrase, 0)
                    + (phrase_occ_trump - phrases_trump.get(phrase, 0))
                )
                * (
                    phrases_obama.get(phrase, 0)
                    + (phrase_occ_obama - phrases_obama.get(phrase, 0))
                )
                * (
                    (phrase_occ_trump - phrases_trump.get(phrase, 0))
                    + (phrase_occ_obama - phrases_obama.get(phrase, 0))
                )
            )
        )
        for phrase in tqdm(distinct_phrases)
    }

    phrase_statistics = pd.DataFrame(
        phrase_statistics.items(), columns=["phrase", "importance"]
    )
    print("eins")

    phrase_statistics["occ_speeches_obama"] = phrase_statistics.apply(
        lambda x: phrases_obama.get(x["phrase"], 0), axis=1
    )
    print("zwei")

    phrase_statistics["occ_speeches_trump"] = phrase_statistics.apply(
        lambda x: phrases_trump.get(x["phrase"], 0), axis=1
    )
    print("drei")

    phrase_statistics["belongs_to"] = phrase_statistics.apply(
        lambda x: "obama"
        if (x["occ_speeches_obama"] > x["occ_speeches_trump"])
        else "trump",
        axis=1,
    )
    print("vier")

    phrase_statistics = phrase_statistics.sort_values(
        "importance", ascending=False, ignore_index=True
    )
    print("fünf")

    return phrase_statistics


def build_ngram_vocab(path_unprocessed_corpus, categories, file_name):
    print("STATUS: START BUILDING NGRAM VOCAB")
    path_base = "data/rhetorical_analysis/"
    path_bigram_vocab = (
        path_base + "-".join(file_name).replace("/", "_") + "-bigram_vocab.pckl"
    )
    path_trigram_vocab = (
        path_base + "-".join(file_name).replace("/", "_") + "-trigram_vocab.pckl"
    )

    if os.path.exists(path_bigram_vocab) and os.path.exists(path_trigram_vocab):
        f = open(path_bigram_vocab, "rb")
        bigram_vocab = pickle.load(f)
        f.close()
        f = open(path_trigram_vocab, "rb")
        trigram_vocab = pickle.load(f)
        f.close()

        print("STATUS: FINISHED BUILDING NGRAM VOCAB (LOADING EXISTING DATA)")
        return (bigram_vocab, trigram_vocab)

    corpus = XMLCorpusReader(root=path_unprocessed_corpus)

    docs = pd.DataFrame(columns=["id", "text"])
    for fileid in corpus.fileids(categories=categories):
        doc = {
            "id": fileid,
            "text": " ".join(list(corpus.paras(fileids=fileid))),
        }
        docs = docs.append(doc, ignore_index=True)

    # remove punctuation
    print("STATUS: REMOVE PUNCTUATION")
    docs["text_processed"] = docs["text"].map(lambda x: re.sub("[,\.!?]", "", x))

    # convert text to lowercase
    print("STATUS: CONVERT TO LOWERCASE")
    docs["text_processed"] = docs["text_processed"].map(lambda x: x.lower())

    # lemmatize
    print("STATUS: LEMMATIZE")
    docs["text_processed"] = lemmatization(
        texts=docs["text_processed"],
        allowed_postags=["NOUN", "ADJ", "VERB", "ADV"],
    )

    # perpare data for ngram
    data = docs.text_processed.values.tolist()
    data_words = list(sent_to_words(data))

    # remove Stop Words
    print("STATUS: REMOVE STOPWORDS")
    data_words_nostops = remove_stopwords(data_words)

    # build bigrams
    print("STATUS: START BUILDING BIGRAM VOCAB")
    phrases = gensim.models.Phrases(data_words, min_count=5, threshold=10)
    bigram_vocab = phrases.vocab

    # build trigrams
    print("STATUS: START BUILDING TRIGRAM VOCAB")
    phrases = gensim.models.Phrases(
        phrases[data_words_nostops], min_count=1, threshold=10
    )
    trigram_vocab = phrases.vocab

    f = open(path_bigram_vocab, "wb")
    pickle.dump(bigram_vocab, f)
    f.close()
    f = open(path_trigram_vocab, "wb")
    pickle.dump(trigram_vocab, f)
    f.close()

    print("STATUS: FINISHED BUILDING NGRAM VOCAB")
    return (bigram_vocab, trigram_vocab)


def get_dep_occ(
    significant_bigrams, significant_trigrams, dep_list, path_unprocessed_corpus
):
    for dep in dep_list:
        col_name = "-".join(dep[0]).replace("/", "_")
        bigram_vocab, trigram_vocab = build_ngram_vocab(
            path_unprocessed_corpus=path_unprocessed_corpus,
            categories=dep[0],
            file_name=dep[0],
        )

        bigram_vocab.update((x, round(y * dep[1])) for x, y in bigram_vocab.items())
        trigram_vocab.update((x, round(y * dep[1])) for x, y in trigram_vocab.items())

        significant_bigrams[col_name] = significant_bigrams.apply(
            lambda x: bigram_vocab.get(x["phrase"], 0), axis=1
        )

        significant_trigrams[col_name] = significant_trigrams.apply(
            lambda x: trigram_vocab.get(x["phrase"], 0), axis=1
        )

    return (significant_bigrams, significant_trigrams)


if __name__ == "__main__":
    PATH_UNPROCESSED_CORPUS = "../corpus"
    PATH_PROCESSED_CORPUS = "../pickled_corpus"
    PATH_PHRASE_STATISTICS_BIGRAMS = (
        "data/rhetorical_analysis/phrase_statistics_bigrams.pckl"
    )
    PATH_PHRASE_STATISTICS_BIGRAMS_DEP = (
        "data/rhetorical_analysis/phrase_statistics_bigrams_dep.pckl"
    )
    PATH_PHRASE_STATISTICS_TRIGRAMS = (
        "data/rhetorical_analysis/phrase_statistics_trigrams.pckl"
    )
    PATH_PHRASE_STATISTICS_TRIGRAMS_DEP = (
        "data/rhetorical_analysis/phrase_statistics_trigrams_dep.pckl"
    )

    ##################################
    ## START ANALYSIS 1
    ##################################
    # get the bigram and trigram vocabulary of the whole corpus except presidential speeches
    corpus = XMLCorpusReader(root=PATH_UNPROCESSED_CORPUS)
    categories = corpus.categories()
    categories.remove("presidential_speeches/obama")
    categories.remove("presidential_speeches/trump")

    bigram_vocab, trigram_vocab = build_ngram_vocab(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
        file_name=["all_except_speeches"],
    )

    # get bigram and trigram vocabulary of the presidential speeches
    categories = ["presidential_speeches/obama"]
    obama_bigram_dict, obama_trigram_dict = build_ngrams(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
        bigram_vocab=bigram_vocab,
        trigram_vocab=trigram_vocab,
    )

    categories = ["presidential_speeches/trump"]
    trump_bigram_dict, trump_trigram_dict = build_ngrams(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories,
        bigram_vocab=bigram_vocab,
        trigram_vocab=trigram_vocab,
    )

    # correct for different number of words between obama and trump
    obama_bigram_dict.update((x, round(y * 3.07)) for x, y in obama_bigram_dict.items())
    obama_trigram_dict.update(
        (x, round(y * 3.07)) for x, y in obama_trigram_dict.items()
    )

    # get final phrase statistics for bigrams and trigrams
    # final bigram model
    if os.path.exists(PATH_PHRASE_STATISTICS_BIGRAMS):
        f = open(PATH_PHRASE_STATISTICS_BIGRAMS, "rb")
        phrase_statistics_bigrams = pickle.load(f)
        f.close()
    else:
        phrase_statistics_bigrams = calc_phrase_statistics(
            phrases_obama=obama_bigram_dict,
            phrases_trump=trump_bigram_dict,
            phrases_corpus=bigram_vocab,
        )
        f = open(PATH_PHRASE_STATISTICS_BIGRAMS, "wb")
        pickle.dump(phrase_statistics_bigrams, f)
        f.close()

    # final trigram model
    if os.path.exists(PATH_PHRASE_STATISTICS_TRIGRAMS):
        f = open(PATH_PHRASE_STATISTICS_TRIGRAMS, "rb")
        phrase_statistics_trigrams = pickle.load(f)
        f.close()
    else:
        phrase_statistics_trigrams = calc_phrase_statistics(
            phrases_obama=obama_trigram_dict,
            phrases_trump=trump_trigram_dict,
            phrases_corpus=trigram_vocab,
        )
        f = open(PATH_PHRASE_STATISTICS_TRIGRAMS, "wb")
        pickle.dump(phrase_statistics_trigrams, f)
        f.close()

    ##################################
    ## START ANALYSIS 3
    ##################################
    """
    excluded_terms = [
        "secretary",
        "obama",
        "trump",
        "Chu",
        "Moniz",
        "Perry",
        "Brouillette",
        "Napolitano",
        "Beers",
        "Johnson",
        "Kelly",
        "Duke",
        "Nielsen",
        "McAleenan",
        "Wolf",
        "Gaynor",
        "Burns",
        "Clinton",
        "Kerry",
        "Shannon",
        "Tillerson",
        "Sullivan",
        "Pompeo",
        "ve_get",
        "re_go",
    ]

    phrase_statistics_bigrams = phrase_statistics_bigrams[
        ~phrase_statistics_bigrams.phrase.str.decode(
            encoding="UTF-8", errors="strict"
        ).str.contains("|".join(excluded_terms))
    ]

    phrase_statistics_trigrams = phrase_statistics_trigrams[
        ~phrase_statistics_trigrams.phrase.str.decode(
            encoding="UTF-8", errors="strict"
        ).str.contains("|".join(excluded_terms))
    ]
    """
    ##################################
    ## END ANALYSIS 3
    ##################################

    # get vocab of departments
    vocab_categories = [
        [["department_of_energy/obama/speeches"], 1.39],
        [
            [
                "department_of_energy/obama/news",
                "department_of_energy/obama/blog_posts",
                "department_of_energy/obama/media_advisories",
            ],
            1,
        ],
        [["department_of_energy/trump/speeches"], 1],
        [
            [
                "department_of_energy/trump/news",
                "department_of_energy/trump/blog_posts",
                "department_of_energy/trump/media_advisories",
            ],
            1.88,
        ],
        [["department_of_homeland_security/obama/speeches"], 1],
        [
            [
                "department_of_homeland_security/obama/press_releases",
                "department_of_homeland_security/obama/blog_posts",
                "department_of_homeland_security/obama/media_advisories",
            ],
            1,
        ],
        [["department_of_homeland_security/trump/speeches"], 1.39],
        [
            [
                "department_of_homeland_security/trump/press_releases",
                "department_of_homeland_security/trump/blog_posts",
                "department_of_homeland_security/trump/media_advisories",
            ],
            1.01,
        ],
        [["department_of_state/obama/remarks_kerry"], 1],
        [["department_of_state/obama/press_releases"], 1],
        [
            [
                "department_of_state/trump/speeches_pompeo",
                "department_of_state/trump/remarks_tillerson",
            ],
            1.31,
        ],
        [["department_of_state/trump/press_releases"], 1.02],
    ]

    for categories in vocab_categories:
        bigram_vocab, trigram_vocab = build_ngram_vocab(
            path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
            categories=categories[0],
            file_name=categories[0],
        )

    # get occurencies in all departments
    significant_bigrams_obama = phrase_statistics_bigrams.query(
        'belongs_to == "obama" and phrase != b"ve_get" and phrase != b"re_go"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:1000]

    significant_trigrams_obama = phrase_statistics_trigrams.query(
        'belongs_to == "obama"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:1000]

    significant_bigrams_obama, significant_trigrams_obama = get_dep_occ(
        significant_bigrams=significant_bigrams_obama,
        significant_trigrams=significant_trigrams_obama,
        dep_list=vocab_categories,
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
    )

    significant_bigrams_trump = phrase_statistics_bigrams.query(
        'belongs_to == "trump"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:1000]

    significant_trigrams_trump = (
        phrase_statistics_trigrams[
            phrase_statistics_trigrams["phrase"]
            .str.decode(encoding="UTF-8", errors="strict")
            .str.contains("donald_trump")
            == False
        ]
        .query('belongs_to == "trump"')
        .sort_values("importance", ascending=False, ignore_index=True)[:1000]
    )

    significant_bigrams_trump, significant_trigrams_trump = get_dep_occ(
        significant_bigrams=significant_bigrams_trump,
        significant_trigrams=significant_trigrams_trump,
        dep_list=vocab_categories,
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
    )

    for vocab_categorie in vocab_categories:
        print(
            "-".join(vocab_categorie[0]).replace("/", "_") + ":",
            significant_trigrams_trump[
                "-".join(vocab_categorie[0]).replace("/", "_")
            ].sum(),
        )
    ##################################
    ## END ANALYSIS 1
    ##################################

    ##################################
    ## START ANALYSIS 2
    ##################################
    """
    corpus = XMLCorpusReader(root=PATH_UNPROCESSED_CORPUS)
    categories = corpus.categories()
    categories.remove("presidential_speeches/obama")
    categories.remove("presidential_speeches/trump")

    categories_obama = [category for category in categories if "obama" in category]
    categories_trump = [category for category in categories if "trump" in category]

    bigram_vocab_dep_obama, trigram_vocab_dep_obama = build_ngram_vocab(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories_obama,
        file_name=["all_dep_obama"],
    )

    bigram_vocab_dep_trump, trigram_vocab_dep_trump = build_ngram_vocab(
        path_unprocessed_corpus=PATH_UNPROCESSED_CORPUS,
        categories=categories_trump,
        file_name=["all_dep_trump"],
    )

    bigram_vocab_dep_obama = {
        k: v for k, v in bigram_vocab_dep_obama.items() if k.count(b"_") == 1
    }
    bigram_vocab_dep_trump = {
        k: v for k, v in bigram_vocab_dep_trump.items() if k.count(b"_") == 1
    }
    trigram_vocab_dep_obama = {
        k: v for k, v in trigram_vocab_dep_obama.items() if k.count(b"_") == 2
    }
    trigram_vocab_dep_trump = {
        k: v for k, v in trigram_vocab_dep_trump.items() if k.count(b"_") == 2
    }

    bigram_vocab_dep_trump.update(
        (x, round(y * 1.12)) for x, y in bigram_vocab_dep_trump.items()
    )

    trigram_vocab_dep_trump.update(
        (x, round(y * 1.12)) for x, y in trigram_vocab_dep_trump.items()
    )

    if os.path.exists(PATH_PHRASE_STATISTICS_BIGRAMS_DEP):
        f = open(PATH_PHRASE_STATISTICS_BIGRAMS_DEP, "rb")
        phrase_statistics_bigrams_dep = pickle.load(f)
        f.close()
    else:
        phrase_statistics_bigrams_dep = calc_phrase_statistics(
            phrases_obama=bigram_vocab_dep_obama,
            phrases_trump=bigram_vocab_dep_trump,
            phrases_corpus=bigram_vocab,
        )
        f = open(PATH_PHRASE_STATISTICS_BIGRAMS_DEP, "wb")
        pickle.dump(phrase_statistics_bigrams_dep, f)
        f.close()

    if os.path.exists(PATH_PHRASE_STATISTICS_TRIGRAMS_DEP):
        f = open(PATH_PHRASE_STATISTICS_TRIGRAMS_DEP, "rb")
        phrase_statistics_trigrams_dep = pickle.load(f)
        f.close()
    else:
        phrase_statistics_trigrams_dep = calc_phrase_statistics(
            phrases_obama=trigram_vocab_dep_obama,
            phrases_trump=trigram_vocab_dep_trump,
            phrases_corpus=trigram_vocab,
        )
        f = open(PATH_PHRASE_STATISTICS_TRIGRAMS_DEP, "wb")
        pickle.dump(phrase_statistics_trigrams_dep, f)
        f.close()

    excluded_terms = [
        b"else_if",
        b"typeof_undefined",
        b"new_array",
        b"function_else",
        b"js_function",
        b"undefined_new",
        b"function_typeof",
        b"getscript_js",
        b"undefined_typeof",
        b"var_function",
        b"runvideo_",
        b"array_getscript",
        b"if_else",
        b"typeof_brightcove",
        b"brightcove_undefined",
        b"if_var",
        b"left_auto",
        b"relative_left",
        b"if_push",
        b"push_runvideo_",
        b"js_function_else",
        b"else_var_function",
        b"function_else_else",
        b"runvideo__var",
        b"var_runvideo_",
    ]

    phrase_statistics_bigrams_dep = phrase_statistics_bigrams_dep[
        ~phrase_statistics_bigrams_dep.phrase.isin(excluded_terms)
    ]

    phrase_statistics_trigrams_dep = phrase_statistics_trigrams_dep[
        ~phrase_statistics_trigrams_dep.phrase.isin(excluded_terms)
    ]

    phrase_statistics_bigrams_dep_obama = phrase_statistics_bigrams_dep.query(
        'belongs_to == "obama"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:60]
    phrase_statistics_bigrams_dep_trump = phrase_statistics_bigrams_dep.query(
        'belongs_to == "trump"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:60]

    phrase_statistics_trigrams_dep_obama = phrase_statistics_trigrams_dep.query(
        'belongs_to == "obama"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:60]
    phrase_statistics_trigrams_dep_trump = phrase_statistics_trigrams_dep.query(
        'belongs_to == "trump"'
    ).sort_values("importance", ascending=False, ignore_index=True)[:60]

    for phrase in phrase_statistics_trigrams_dep_trump[30:60]["phrase"]:
        print(phrase.decode(encoding="UTF-8", errors="strict"))
    """
    ##################################
    ## END ANALYSIS 2
    ##################################
