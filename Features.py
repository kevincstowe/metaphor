import Corpus
import numpy as np
from operator import add

import SDP

def bow(word, **params):
    if "lex" not in params:
        raise Exception("BOW feature needs a 'lex' feature - the dictionary to index words from")
    if "field" not in params:
        raise Exception("BOW feature needs a 'field' feature - what are making a bag of?")

    field = params["field"]
    res = [0] * len(params["lex"])

    prev_context = params["prev_context"] if "prev_context" in params else 0
    next_context = params["prev_context"] if "next_context" in params else 0

    words = []
    sent_ind = word.sentence.words.index(word)

    if "sentence_context" in params and params["sentence_context"]:
        words = [getattr(w, field) for w in word.sentence.words]
    else:
        for i in range(sent_ind-prev_context, sent_ind+next_context):
            if i < 0 or i >= len(word.sentence.words):
                pass
            else:
                words.append(getattr(word.sentence.words[i], field))

    for item in words:
        if item in params["lex"]:
            res[params["lex"].index(item)] += 1
    return res

def continuous(word, **params):
    if "model" not in params:
        raise Exception("Continuous featurization requires a model")
    if "field" not in params:
        raise Exception("Continuous featurization requires a field")

    res = []
    model = params["model"]
    field = params["field"]
    prev_context = params["prev_context"] if "prev_context" in params else 0
    next_context = params["prev_context"] if "next_context" in params else 0
    average = params["average"] if "average" in params else False

    if "sentence_context" in params and params["sentence_context"]:
        words = [w for w in word.sentence.words if w != word]
        for item in words:
            data = model[item] if item and item in model else [0]*model.vector_size
            res = map(add, res, data)
    else:
        words = []
        sent_ind = word.sentence.words.index(word)
        for i in range(sent_ind-prev_context, sent_ind+next_context+1):
            if i < 0 or i >= len(word.sentence.words):
                words.append(None)
            else:
                words.append(getattr(word.sentence.words[i], field))
        for item in words:
            data = model[item] if item and item in model else [0]*model.vector_size
            res.extend(data)
    if average:
        return [val/len(res) for val in res]
    return res

def frame(word, **params):
    if "fp" not in params:
        raise Exception("Need a frame predictor: load FramePredictor.FramePredictor()")

    fp = params["fp"]
    sent_text = word.sentence.text()

    clauses = SDP.trim_clauses(sent_text)
    for c in clauses:
        pred = fp.predict(c[1], vnc=word.vnc)
        print (c, pred)