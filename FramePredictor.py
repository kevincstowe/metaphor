import sys
import json
import numpy as np
import re
import random

from sklearn.ensemble import RandomForestClassifier

import SDP
import ML

local_verbnet_api_path = "/home/kevin/GitHub/verbnet/api/"
sys.path.append(local_verbnet_api_path)

import verbnet

class FramePredictor(object):
    words, poss, rels = [], [], []

    def __init__(self):
        vn = verbnet.VerbNetParser(version="3.3")
        self.parse_frames(vn, load=False, secondary_information=True)
        self.generate_samples()

        for key, parse_key, parse in self.samples:
            for word in parse:
                if len(word) == 4:
                    if word[0].lower() not in self.words:
                        self.words.append(word[0].lower())
                    if word[1] not in self.poss:
                        self.poss.append(word[1])
                    if word[3] not in self.rels:
                        self.rels.append(word[3])

        self.model = self.train_model()

    def train_model(self):
        X, y = self.vectorize_samples(self.samples)
        return ML.build_model(X, y, model_type=RandomForestClassifier, params={"n_estimators": 1000})

    def predict(self, parse, vnc=None):
        if len(parse[0]) != 4:
            parse = [[p[1], p[3], p[6], p[7]] for p in parse]
        v = self.vectorize_parse(parse)
        v = np.array(v).reshape(1, -1)

        # If we have a VNC, we can constrain frames to only those in that class
        if vnc:
            pred = self.model.predict_proba(v)[0]
            try:                        # constrained prediction
                pred = max([[self.model.classes_[j], pred[j]] for j in range(len(pred)) if
                              pred[j] and self.model.classes_[j] in self.frame_parses[vnc]], key=lambda x: x[1])[0]
            except Exception as e:     # no frames in the class that match the prediction
                pred = self.model.classes_[np.argmax(pred)]
        else:
            pred = self.model.predict(v)[0]
        return pred

    def generate_samples(self):
        all_classes = list(self.frame_parses.keys())
        samples = []
        for c in all_classes:
            for parse_key in self.frame_parses[c]:
                for parse in self.frame_parses[c][parse_key]:
                    samples.append((c, parse_key, parse))
        self.samples = samples

    def run_ml(self, split=.5):
        random.shuffle(self.samples)
        train_samples = self.samples[:int(len(self.samples)*split)]
        test_samples = self.samples[int(len(self.samples)*split):]

        train_X, train_y = self.vectorize_samples(train_samples)
        test_X, test_y = self.vectorize_samples(test_samples)

        for estimators in [1000]:
            model = ML.build_model(train_X, train_y, model_type=RandomForestClassifier, params={"n_estimators": estimators})
            predictions = model.predict_proba(test_X)

            correct = 0
            for i in range(len(test_y)):
                poss_frames = list(self.frame_parses[test_samples[i][0]].keys())
                try:
                    y_pred = max([[model.classes_[j], predictions[i][j]] for j in range(len(predictions[i])) if predictions[i][j] and model.classes_[j] in poss_frames], key=lambda x : x[1])[0]
                except ValueError as e: #no frames in the class that match the prediction
                    y_pred = model.classes_[np.argmax(predictions[i])]
                if test_y[i] == y_pred:
                    correct += 1

            print(correct*1. / len(test_y))

    def vectorize_parse(self, parse):
        vec = [0] * (len(self.words) + len(self.poss) + len(self.rels))
        for i in range(len(parse)):
            word = parse[i]
            if len(word) == 4:
                if word[0].lower() in self.words:
                    vec[self.words.index(word[0].lower())] += 1
                if word[1] in self.poss:
                    vec[len(self.words) + self.poss.index(word[1])] += 1
                if word[3] in self.rels:
                    vec[len(self.words) + len(self.poss) + self.rels.index(word[3])] += 1
        return vec

    def vectorize_samples(self, parse_samples):
        X, y = [], []
        for key, parse_key, parse in parse_samples:
            X.append(self.vectorize_parse(parse))
            y.append(parse_key)
        X = np.array(X)
        y = np.array(y)
        return X, y

    def parse_frames(self, vn, load=True, secondary_information=False, filename="vn_frame_parses.json"):
        if load:
            self.frame_parses = json.load(open(filename))
        else:
            res = {}
            for vnc in vn.get_verb_classes():
                if vnc.numerical_ID not in res:
                    res[vnc.numerical_ID] = {}
                for frame in vnc.frames:
                    # take out secondary information
                    if not secondary_information:
                        label_data = [re.split("[._-]", e)[0] for e in frame.primary if "recipient" not in e and "topic" not in e and "attribute" not in e]
                    else:
                        label_data = frame.primary
                    # condense wh- words
                    label_data = " ".join(["WH" if w.startswith("wh") else w for w in label_data])

                    for e in frame.examples:
                        parse = [item.split("\t") for item in list(SDP.parse_sentence(e))[0].to_conll(style=4).split("\n")]
                        if label_data not in res[vnc.numerical_ID]:
                            res[vnc.numerical_ID][label_data] = []
                        res[vnc.numerical_ID][label_data].append(parse)
            json.dump(res, open(filename, "w"))
            self.frame_parses = res


def get_corpus_frames(corpus):
    fp = FramePredictor()
    data = {}
    count = 0
    for sentence in corpus.instances:
        clauses = SDP.trim_clauses(sentence.text())
        count += 1
        for c in clauses:
            for w in sentence.words:
                if w.text == c[0][1]:
                    if w.lemma not in data:
                        data[w.lemma] = {"L":{k:0 for k in fp.model.classes_}, "M":{k:0 for k in fp.model.classes_}}

                    pred = fp.predict(c[1], vnc=w.vnc)
                    if w.met != "N":
                        if pred not in data[w.lemma]["M"]:
                            data[w.lemma]["M"][pred] = 0
                        data[w.lemma]["M"][pred] += 1
                    else:
                        if pred not in data[w.lemma]["L"]:
                            data[w.lemma]["L"][pred] = 0
                        data[w.lemma]["L"][pred] += 1

    #print all the possible frames
    print ("-," + ",".join([cl for cl in sorted(fp.model.classes_)]))


    #sum up the counts for each frame
    l_frame_sums = []
    m_frame_sums = []
    for k in sorted(fp.model.classes_):
        l_frame_sums.append(sum([data[key]["L"][k] for key in data.keys()]))
        m_frame_sums.append(sum([data[key]["M"][k] for key in data.keys()]))

    #print sum of the literal and metaphor counts by frame
    print (str(sum(l_frame_sums)) + "," + ",".join([str(s) for s in l_frame_sums]))
    print (str(sum(m_frame_sums)) + "," + ",".join([str(s) for s in m_frame_sums]))

    #sort keys by their total counts, then print their sums and counts by frame
    for key in sorted(list(data.keys()), key=lambda x: sum(data[x]["L"].values()) + sum(data[x]["M"].values()), reverse=True):
        print (key)
        print (str(sum(data[key]["L"].values())) + "," + ",".join([str(data[key]["L"][v]) for v in sorted(fp.model.classes_)]))
        print (str(sum(data[key]["M"].values())) + "," + ",".join([str(data[key]["M"][v]) for v in sorted(fp.model.classes_)]))


if __name__ == "__main__":
    random.seed(100)
    fp = FramePredictor()
#    fps = json.load(open("vn_frame_parses.json"))
    fps = fp.frame_parses
    classes = ["37.1.1", "37.1.1-1", "37.1.1-1-1",
               "37.1.2",
               "37.1.3",
               "37.2",
               "37.3",
               "37.4.1",
               "37.4.2", "37.4.2-1",
               "37.5",
               "37.6", "37.6-1",
               "37.7", "37.7-1", "37.7-1-1", "37.7-1-1-1", "37.7-1-2",
               "37.8",
               "37.9", "37.9-1",
               "37.10",
               "37.11", "37.11-1", "37.11-1-1", "37.11-2",
               "37.12",
               "37.13"]
    all_frames = set()
    for cl in classes:
        for k in fps[cl]:
            all_frames.add(k)

    all_frames = sorted(list(all_frames))
    print (";;".join([""] + all_frames))
    for cl in classes:
        cur_frames = [0]*len(all_frames)
        for k in fps[cl]:
            cur_frames[all_frames.index(k)] += 1
        print (";;".join([cl] + [str(s) for s in cur_frames]))

#    for i in range(10):
#        fp.run_ml(split=.8)

    #get_corpus_frames(VUAMCLoader.VUAMCCorpus())
