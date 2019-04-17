import sys
import json

sys.path.append("../")
import VUAMCLoader

import numpy as np
import time
import cProfile

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score

from gensim.corpora import dictionary
from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences

from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Concatenate

from keras.models import Model

from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Activation

from keras.optimizers import SGD

GOOGLE_NEWS_VECTORS = "C:/Users/Kevin/PycharmProjects/vectors/models/GoogleNews-vectors-negative300.bin"

def words_to_ids(words, word2id, prev_context, next_context):
    ids = []
    for word in words:
        word_ids = []
        for i in range(word.index - prev_context, word.index + next_context + 1, 1):
            if i >= 0 and i < len(word.sentence.words):
                word_ids.append(word2id[word.sentence.words[i].lemma])
            else:
                word_ids.append(word2id["NULL"])
        ids.append(word_ids)
    return np.array(ids)

def pos_to_ids(words, pos2id, prev_context, next_context):
    ids = []
    for word in words:
        pos_ids = []
        for i in range(word.index - prev_context, word.index + next_context + 1, 1):
            if i >= 0 and i < len(word.sentence.words):
                pos_ids.append(pos2id[word.sentence.words[i].pos])
            else:
                pos_ids.append(pos2id["NULL"])
        ids.append(pos_ids)
    return np.array(ids)

def get_embeddings(word2iddict, model=None):
    if not model:
        model = KeyedVectors.load_word2vec_format(GOOGLE_NEWS_VECTORS, binary=True)
    res = [[0] * 300] * len(word2iddict)

    for word in word2iddict:
        if word in model:
            res[word2iddict[word]] = model[word]
    return np.array(res)


class NN(object):
    def __init__(self,  corpus, pc=2, nc=2, maxlen=50, verbs=False):
        self.corpus = corpus
        self.context_size = pc + nc + 1
        self.model = None
        self.maxlen = maxlen

        id2word = dictionary.Dictionary([[w.lemma for w in corpus.words]])
        pos2word = dictionary.Dictionary([[w.pos for w in corpus.words]])

        word2id = {y: x for x, y in id2word.items()}
        word2id["NULL"] = len(word2id)
        pos2id = {y: x for x, y in pos2word.items()}
        pos2id["NULL"] = len(pos2id)

        self.word2id = word2id
        self.pos2id = pos2id

        self.initialize_word_data(verbs, pc, nc)
        #self.initialize_sentence_data(verbs)


    def initialize_sentence_data(self, verbs):
        train_sents = self.corpus.get_training_sents()
        test_sents = self.corpus.get_test_sents()

        lem_train_X = [[self.word2id[w.lemma] for w in s.words] for s in train_sents]
        self.sent_lemma_train_X = pad_sequences(maxlen=self.maxlen, sequences=lem_train_X, padding="post")
        lem_test_X = [[self.word2id[w.lemma] for w in s.words] for s in test_sents]
        self.sent_lemma_test_X = pad_sequences(maxlen=self.maxlen, sequences=lem_test_X, padding="post")

        train_y = [[[0,1] if "met" in w.met else [1,0] for w in s.words] for s in train_sents]
        self.sent_train_y = pad_sequences(maxlen=self.maxlen, sequences=train_y, padding="post", value=[0,0])
        test_y = [[[0,1] if "met" in w.met else [1,0] for w in s.words] for s in test_sents]
        self.sent_test_y = pad_sequences(maxlen=self.maxlen, sequences=test_y, padding="post", value=[0,0])


    def initialize_word_data(self, verbs, pc, nc):
        if verbs:
            training_data = corpus.get_verb_training_data()
            test_data = corpus.get_verb_test_data()
        else:
            training_data = corpus.get_training_data()
            test_data = corpus.get_test_data()

        self.lemma_train_x = words_to_ids(training_data, self.word2id, prev_context=pc, next_context=nc)
        self.pos_train_x = pos_to_ids(training_data, self.pos2id, prev_context=pc, next_context=nc)
        self.lemma_test_x = words_to_ids(test_data, self.word2id, prev_context=pc, next_context=nc)
        self.pos_test_x = pos_to_ids(test_data, self.pos2id, prev_context=pc, next_context=nc)
        self.train_y = [[0,1] if "met" in w.met else [1,0] for w in training_data]
        self.test_y = [[0,1] if "met" in w.met else [1,0] for w in test_data]

    def evaluate(self, test_data):
        preds = self.model.predict(test_data)

        ys = [item.index(max(item)) for item in self.test_y]
        preds = [list(item).index(max(list(item))) for item in list(preds)]
        return f1_score(ys, preds), precision_score(ys, preds), recall_score(ys, preds)

    def lstm_evaluate(self):
        preds = self.model.predict([self.sent_lemma_test_X])

        ys = [[np.argmax(item) for item in sentence if sum(item)] for sentence in self.sent_test_y]
        preds = [[np.argmax(preds[j][i]) for i in range(len(ys[j]))] for j in range(len(preds))]
        ys = [item for sublist in ys for item in sublist]
        preds = [item for sublist in preds for item in sublist]

        return f1_score(ys, preds), precision_score(ys, preds), recall_score(ys, preds)

    def train_cnn(self, embeddings, dropout=.5):
        lemma_input = Input(shape=[self.context_size], dtype='float')
        pos_input = Input(shape=[self.context_size], dtype='float')

        b1 = Embedding(len(self.word2id), len(embeddings[0]), weights=[embeddings])(lemma_input)
        c1 = Conv1D(filters=16, kernel_size=2, padding='same', activation='tanh')(b1)
        p1 = MaxPooling1D(pool_size=4, padding='same')(c1)

        pb1 = Embedding(len(self.word2id), len(embeddings[0]), weights=[embeddings])(pos_input)
        pc1 = Conv1D(filters=16, kernel_size=2, padding='same', activation='tanh')(pb1)
        pp1 = MaxPooling1D(pool_size=4, padding='same')(pc1)

        c = Concatenate()([p1, pp1])

        do = Dropout(dropout)(c)
        fl = Flatten()(do)

        dense_out = Dense(2, activation='softmax')(fl)

        model = Model(inputs=[lemma_input, pos_input], outputs=dense_out)
        model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['categorical_accuracy'])

        model.fit([self.lemma_train_x, self.pos_train_x], self.train_y, epochs=25, batch_size=1024, verbose=0)

        self.model = model

    def train_mlp(self, embeddings, dropout=.75, nodes=25, layers=1, loss="mean_squared_error", lr=.1, momentum=0., decay=0.):
        lemma_input = Input(shape=[self.context_size], dtype='float')
        pos_input = Input(shape=[self.context_size], dtype='float')

        b1 = Embedding(len(self.word2id), len(embeddings[0]), weights=[embeddings], trainable=True)(lemma_input)
        b2 = Embedding(len(self.pos2id), 10, trainable=True)(pos_input)

        c = Concatenate()([b1, b2])

        d1 = Dense(nodes, activation="tanh")(c)
        d = Dropout(dropout)(d1)
        f = Flatten()(d)
        o = Dense(2, activation="softmax")(f)

        model = Model(inputs=[lemma_input, pos_input], outputs=o)
        model.compile(loss=loss, optimizer=SGD(lr=lr, momentum=momentum, decay=decay), metrics=['categorical_accuracy'])

        model.fit([self.lemma_train_x, self.pos_train_x], self.train_y, epochs=25, batch_size=1024, verbose=1)

        self.model = model

    def train_lstm(self, dropout):
        return

if __name__ == "__main__":
    np.random.seed(1337)
    corpus = VUAMCLoader.VUAMCCorpus()

    neural = NN(corpus)
    embeddings = get_embeddings(neural.word2id)

    neural.train_mlp(embeddings)
    neural.evaluate([neural.lemma_test_x, neural.pos_test_x])