import numpy as np
import tqdm
import gensim
import pandas as pd
import pyLDAvis.gensim
import pickle
import pyLDAvis
import os

from gensim.models import CoherenceModel


class LDATester:
    grid = {}
    grid["Validation_Set"] = {}

    def __init__(self, corpus, texts, id2word):
        self.corpus = corpus
        self.texts = texts
        self.id2word = id2word

    def compute_coherence_values(self, corpus, dictionary, k, a, b):
        lda_model = gensim.models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            random_state=100,
            chunksize=100,
            passes=10,
            alpha=a,
            eta=b,
        )

        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=self.texts, dictionary=self.id2word, coherence="c_v"
        )

        return coherence_model_lda.get_coherence()

    def opt_lda(self, file_name, min_topics, max_topics, step_size, alpha, beta):
        print("STATUS: START OPTIMIZING LDA")

        topics_range = range(min_topics, max_topics + step_size, step_size)
        n_docs = len(self.corpus)
        corpus_sets = [
            # gensim.utils.ClippedCorpus(self.corpus, round(n_docs * 0.75)),
            self.corpus,
        ]

        corpus_title = ["100% Corpus"]

        model_results = {
            "Validation_Set": [],
            "Topics": [],
            "Alpha": [],
            "Beta": [],
            "Coherence": [],
        }

        # Can take a long time to run
        if 1 == 1:
            num_of_iterations = (
                len(range(len(corpus_sets)))
                * len(topics_range)
                * len(alpha)
                * len(beta)
            )
            pbar = tqdm.tqdm(total=num_of_iterations)

            # iterate through validation corpuses
            for i in range(len(corpus_sets)):
                # iterate through number of topics
                for k in topics_range:
                    # iterate through alpha values
                    for a in alpha:
                        # iterare through beta values
                        for b in beta:
                            # get the coherence score for the given parameters
                            cv = self.compute_coherence_values(
                                corpus=corpus_sets[i],
                                dictionary=self.id2word,
                                k=k,
                                a=a,
                                b=b,
                            )
                            # Save the model results
                            model_results["Validation_Set"].append(corpus_title[i])
                            model_results["Topics"].append(k)
                            model_results["Alpha"].append(a)
                            model_results["Beta"].append(b)
                            model_results["Coherence"].append(cv)

                            pbar.update(1)

            pbar.close()

            pd.DataFrame(model_results).to_excel(
                "./persistent_computations/lda_tuning_results-" + file_name + ".xlsx",
                index=False,
            )
            print("STATUS: FINISHED OPTIMIZING LDA")

    def compute_lda(self, file_name, n_topics, alpha, beta):
        print("STATUS: START BUILDING MODEL")
        path_base = "persistent_computations/"
        path_lda_model = path_base + file_name + "-lda_model.pckl"
        if os.path.exists(path_lda_model):
            f = open(path_lda_model, "rb")
            lda_model = pickle.load(f)
            f.close()
            print("STATUS: FINISHED BUILDING MODEL (USING ALREADY BUILT MODEL)")
        else:
            lda_model = gensim.models.LdaMulticore(
                corpus=self.corpus,
                id2word=self.id2word,
                num_topics=n_topics,
                random_state=100,
                chunksize=100,
                passes=10,
                alpha=alpha,
                eta=beta,
            )
            f = open(path_lda_model, "wb")
            pickle.dump(lda_model, f)
            f.close()
            print("STATUS: FINISHED BUILDING MODEL (NEW MODEL CREATED)")

        # Visualize the topics
        print("STATUS: START VISUALIZING MODEL")
        data = pyLDAvis.gensim.prepare(lda_model, self.corpus, self.id2word)
        pyLDAvis.show(data)
