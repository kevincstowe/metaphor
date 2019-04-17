from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

from nltk.corpus import stopwords

import pickle
import numpy as np

import ML
import CorpusLoaders
import Features
import FramePredictor

from gensim.models import KeyedVectors

GOOGLE_NEWS_VECTORS = "/home/kevin/vectors/GoogleNews-vectors-negative300.bin"


class Learner(object):
    def __init__(self, corpus, pc=1, nc=1, verbs=True, words=True, sentences=True, size=0):
        self.corpus = corpus
        self.pc = pc
        self.nc = nc
        self.context_size = pc + nc + 1
        self.model = None
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []

        self.training_data = self.corpus.get_training_data()
        if size:
            self.training_data = self.training_data[:size]

        self.train_lemma_lex = list(set([w.lemma for w in self.training_data if w.lemma not in stopwords.words(["english"])]))
        self.construction_lex = [w.construction for w in self.training_data]

    def featurize_dataset(self, embeddings=None, frame_predictor=None):
        train_x, test_x, train_y, test_y = [], [], [], []

        frame_predictor = FramePredictor.FramePredictor()
        features = [(Features.bow, {"lex":self.train_lemma_lex, "field":"lemma", "prev_context":self.pc, "next_context":self.nc}),
                    (Features.continuous, {"model": embeddings, "prev_context": self.pc, "next_context": self.nc, "field": "lemma"}),
                    (Features.continuous, {"model": embeddings, "sentence_context":True, "field":"lemma"})
#            (Features.frame, {"fp":frame_predictor})
                    ]

        for word in self.training_data:
            train_x.append(featurize(word, features))
            train_y.append(featurize_y(word))

        for word in self.corpus.get_test_data():
            test_x.append(featurize(word, features))
            test_y.append(featurize_y(word))

        self.train_x = np.array(train_x)
        self.train_y = train_y
        self.test_x = np.array(test_x)
        self.test_y = test_y

    def train_model(self, model_type, params, save="testing.model"):
        m = ML.build_model(self.train_x, self.train_y, model_type, params)
        if save:
            pickle.dump(m, open(save, "wb"))
        self.model = m

    def evaluate(self):
        return (ML.evaluate(self.test_x, self.test_y, self.model))


def featurize(word, features):
    res = []
    for feat in features:
        res.extend(feat[0](word, **feat[1]))
    return res


def featurize_y(word):
    if "met" not in word.met:
        return 0
    return 1


if __name__ == "__main__":
    size = 0
    corpus = CorpusLoaders.VUAMCCorpus()
    ml = Learner(corpus, size=size)
    #embeddings = KeyedVectors.load_word2vec_format(GOOGLE_NEWS_VECTORS, binary=True)

    for con in [1]:
        ml.pc = con
        ml.nc = con
        ml.featurize_dataset()
        ml.train_model(model_type=BernoulliNB, params={})
        print (ml)
        print (ml.evaluate())
        
        '''
        for e in [20, 50, 100, 200, 500]:
            for mf in ["auto", "sqrt", "log2"]:
                for md in [50, 200, None]:
                    for model_type in [RandomForestClassifier]:
                        ml.pc = con
                        ml.nc = con
                        ml.train_model(model_type=model_type, params={"n_estimators":e, "max_features":mf, "max_depth":md})

                        res = ml.evaluate()
                        print (str((res, con, e, mf, md)))
                        with open("ml_results", "a") as output:
                            output.write(str((res, con, e, mf, md, size)) + "\n")
        '''
