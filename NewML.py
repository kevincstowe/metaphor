from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
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

GLOVE_VN_VECTORS = "/home/kevin/glove/glove_models/glove-sense450.w"


CONCRETENESS_PATH = "/home/kevin/metaphor/corpora/concreteness.txt"
ANEW_PATH = "/home/kevin/metaphor/corpora/anew.csv"
MRC_PATH = "/home/kevin/metaphor/corpora/1054/mrc2.dct"

concrete_dict = {}
anew_dict = {}
mrc_dict = {}

CLASSES = ["met", "target", "source", "tag-1"]

#ANEW mins 1.26, 1.6, 1.68
#ANEW maxs 8.53, 7.79, 7.9
#ANEW mid 4.6 4.7 4.7sh??

def initialize_dicts():
    global concrete_dict
    global anew_dict
    global mrc_dict
    
    for line in open(CONCRETENESS_PATH).readlines()[1:-1]:
        concrete_dict[line.split()[0]] = float(line.split()[2])

    for line in open(ANEW_PATH).readlines()[1:]:
        data = line.split(",")
        anew_dict[data[1]] = (float(data[2]), float(data[5]), float(data[8]))
        
    for line in open(MRC_PATH).readlines():
        if int(line[31:34]):
            mrc_dict[line[51:].split("|")[0].lower()] = int(line[31:34]) / 100
            

def convert_met(word, classes=CLASSES, multi=False):
    if not multi:
        for c in classes:
            if c in word.met:
                return 1
        return 0
    else:
        res = 0
        if word.met[1] == "":
            pass
        elif word.met[1] == []:
            res = 1
        elif type(word.met[1]) == list:
            if word.met[1][0] in CorpusLoaders.DOMAINS:
                res = CorpusLoaders.DOMAINS.index(word.met[1][0])
            else:
                print ("target domain didn't match: " + word.met[1][0])
        else:
            for source_d in word.met[1]:
                if source_d[1] > 2:
                    if source_d[0] in CorpusLoaders.DOMAINS:
                        res = CorpusLoaders.DOMAINS.index(source_d[0])
            res = 0
        return res

def convert_word_to_features(word, features={}):
    res = {}
    if "bigram" in features and word.index > 0:
        res[str(word.sentence.words[word.index-1].lemma) + "-" + word.lemma] = 1

    if "trigram" in features and word.index > 1:
        res[str(word.sentence.words[word.index-2].lemma) + "-" + str(word.sentence.words[word.index-1].lemma) + "-" + word.lemma] = 1

    context_words = []

    for i in range(-features["pre"], features["nex"]+1, 1):
        if word.index + i >= 0 and word.index + i < len(word.sentence.words):
            if i > 0:
                prefix = "nex-"
            elif i < 0:
                prefix = "pre-"
            else:
                prefix = "cur-"
            context_words.append((word.sentence.words[word.index+i], prefix))

        
    if "head" in features:
        head_word = word.sentence.find_head(word)
        if head_word:
            context_words.append((head_word, "head-"))

    if "deps" in features:
        deps = word.sentence.find_dependencies(word)
        for d in deps:
            context_words.append((d, "deps-"))
            
    for w in context_words:
        cur_word, prefix = w
        res[prefix + "lemma-" +cur_word.lemma] = 1
                    
        if "pos" in features:
            if cur_word.pos:
                res[prefix + "pos-"+ cur_word.pos] = 1
        
        if "anew" in features:
            if cur_word.lemma in anew_dict:
                res[prefix + "valence"], res[prefix + "arousal"], res[prefix + "dominance"] = anew_dict[cur_word.lemma]
            else:
                res[prefix + "valence"], res[prefix + "arousal"], res[prefix + "dominance"] = 4.75, 4.75, 4.75
            
        if "concrete" in features:
            res[prefix + "concreteness"] = concrete_dict[cur_word.lemma] if cur_word.lemma in concrete_dict else 3

        if "mrc" in features:
            res[prefix + "mrc"] = mrc_dict[cur_word.lemma] if cur_word.lemma in mrc_dict else 4

        if "embeddings" in features:
            cur_emb = [0] * 300
            if cur_word.lemma in features["embeddings"]:
                cur_emb = features["embeddings"][cur_word.lemma]
            for i in range(len(cur_emb)):
                if prefix + "-" + str(i) not in res:
                    res[prefix + "-" + str(i)] = 0
                res[prefix + "-" + str(i)] += cur_emb[i]

        if "tag" in features:
            if prefix != "cur-":
                if prefix + "tag" not in res:
                    res[prefix + "tag"] = 0
                res[prefix + "tag"] += convert_met(cur_word)

        if "vn_embed" in features:
            cur_emb = [0] * 100
            if cur_word.vnc and cur_word.text + "_" + cur_word.vnc in features["vn_embed"]:
                cur_emb = features["vn_embed"][cur_word.text + "_" + cur_word.vnc]
            for i in range(len(cur_emb)):
                if prefix + "-vn-" + str(i) not in res:
                    res[prefix + "-vn-" + str(i)] = 0
                res[prefix + "-vn-" + str(i)] += cur_emb[i]
                
    return res


def train_dv(training_data, feature_set, multi=False):
    cat_features = []
    con_features = []
    y = []

    for word in training_data:
        new_cons = convert_word_to_features(word, feature_set)
        con_features.append(new_cons)
        y.append(convert_met(word, multi=multi))

    con_dvx = DictVectorizer(sparse=True)

    con_features = con_dvx.fit_transform(con_features)

    X = con_features
    
    return X, y, con_dvx


def featurize(dataset, con_dvx, feature_set):
    #cat_feats = []
    con_feats = []
    
    for word in dataset:
        new_cons = convert_word_to_features(word, feature_set)
#        cat_feats.append(new_cats)
        con_feats.append(new_cons)

#    new_cats = cat_dvx.transform(cat_feats)
    new_cons = con_dvx.transform(con_feats)
#    new_cons = scaler.transform(new_cons)

#    res = np.concatenate((new_cats, new_cons), axis = 1)
    res = new_cons
#    print (np.array(res).shape)
                   
    return res


class Learner(object):
    def __init__(self, corpus, size=None, verb=False, cv=False):
        self.corpus = corpus
        self.model = None
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []
        if cv:
            self.training_data = self.corpus.get_training_data()
        else:
            if verb:
                self.training_data = self.corpus.get_verb_training_data()
                self.test_data = self.corpus.get_verb_test_data()
            else:
                self.training_data = self.corpus.get_training_data()
                self.test_data = self.corpus.get_test_data()
                
        if size:
            self.training_data = self.training_data[:size]

    def cv(self, feature_set, model_type, params, folds=5, multi=False):
        train_x = self.training_data
        train_y = [convert_met(word, multi=multi) for word in self.training_data]
        train_x, _, con_dvx = train_dv(train_x, feature_set)
        
        return ML.run_cv(train_x, train_y, model_type, params, folds)
            
    def featurize_dataset(self, feature_set, multi=False):
        train_x, train_y, con_dvx = train_dv(self.training_data, feature_set, multi)
        test_x, test_y = [], []
        
        test_x = featurize(self.test_data, con_dvx, feature_set)
        test_y = [convert_met(word, multi=multi) for word in self.test_data]

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def train_model(self, model_type, params, save="testing.model"):
        m = ML.build_model(self.train_x, self.train_y, model_type, params)
        if save:
            pickle.dump(m, open(save, "wb"))
        self.model = m

    def evaluate(self, multi):
        if multi:
            return ML.multi_evaluate(self.test_x, self.test_y, self.model)
        else:
            return (ML.evaluate(self.test_x, self.test_y, self.model))


def featurize_y(word):
    if "met" not in word.met:
        return 0
    return 1


if __name__ == "__main__":
    size = 0
    print ("initializing dicts...")
    initialize_dicts()
            
    print ("loading corpus...")
    corpus = CorpusLoaders.VUAMCCorpus()
    cv = False
    multi = False
    verb = False
    
    ml = Learner(corpus, size=size, verb=verb, cv=cv)
    embeddings = KeyedVectors.load_word2vec_format(GOOGLE_NEWS_VECTORS, binary=True)
    vn_glove = KeyedVectors.load_word2vec_format(GLOVE_VN_VECTORS)
    
    print ("featurizing data...")

    for m in [(LogisticRegression, {"solver":"liblinear", "C":1}),
              (BernoulliNB, {"alpha":.001}),
              (LinearSVC, {"max_iter":100000})
              ]:
        for feature_set in [{"pre":2, "nex":2, "bigram":1, "anew":1, "concrete":1, "mrc":1, "pos":1, "embeddings":embeddings, "vn_embed":vn_glove}]:

#        for feature_set in [{"pre":2, "nex":2, "vn_embed":vn_glove}]:
#                            None,
#                            {"bigram":0, "pre":0, "nex":0},
#                            {"trigram":1, "pre":0, "nex":0},
#                            {"pos":1, "pre":0, "nex":0},
#                            None,
#                            {"pre":1, "nex":0},
#                            {"pre":2, "nex":0},
#                            {"pre":3, "nex":0},
#                            None,
#                            None,
#                            {"pre":0, "nex":1},
#                            {"pre":0, "nex":2},
#                            {"pre":0, "nex":3},
#                            None,
#                            None,
#                            {"pre":1, "nex":1},
#                            {"pre":2, "nex":2},
#                            {"pre":3, "nex":3},
#                            None,
#                            {"pre":1, "nex":1, "anew":0},
#                            {"pre":1, "nex":1, "concrete":0},
#                            {"pre":1, "nex":1, "mrc":0},
#                            {"pre":1, "nex":1, "mrc":0, "anew":0, "concrete":0},
#                            None,
#                            {"pre":2, "nex":2, "mrc":1, "anew":1, "concrete":1, "pos":1, "bigram":1}]:

            if not feature_set:
                print ("-")
            else:
                if cv:
                    print(ml.cv(feature_set, model_type=m[0], params=m[1], multi=multi))
                else:
                    ml.featurize_dataset(feature_set, multi=multi)
                    ml.train_model(model_type=m[0], params=m[1])
                    print (ml.evaluate(multi))
        print ("---")
