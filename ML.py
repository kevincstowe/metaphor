from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score
from math import sqrt

import random

def run_cv(X, y, model_type, params, folds=5):
    classifier = model_type(**params)
    scores = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(f1_score))
    #scores = cross_val_score(classifier, X, y, cv=folds, scoring="f1")
    return (sum(scores)/folds)


def f1(v_data):
    prec, rec, f1 = 0,0,0
    if sum(v_data[1]) > 0:
        prec = v_data[1][1] / sum(v_data[1])
    if v_data[1][1] + v_data[0][1]:
        rec = v_data[1][1] / (v_data[1][1] + v_data[0][1])


    if prec * rec > 0:
        f1 = 2 * ((prec * rec) / (prec + rec))
    return f1

def fivetwo(X, y, model_type, params):
    res = []

    for i in range(5):
#        combined = list(zip(X, y))
#        random.shuffle(combined)
#        X, y = zip(*combined)
#        classifier = model_type(**params)
#        print (X)
#        classifier.fit(X, y)
        classifier = model_type(**params).fit(X, y)
        preds = classifier.predict(X[int(len(X)/2):])

        print (preds, y[int(len(y)/2)])
        print (res)

def replacement(X, y, model, n=1000):
    predictions = model.predict(X)
    f1s = []
    for i in range(n):
        res = [[0,0],[0,0]]
        for j in range(len(predictions)):
            index = random.randrange(0, len(predictions)-1)
            res[predictions[index]][y[index]] += 1
        f1s.append(f1(res))
    mean = sum(f1s) / len(f1s)
    std = sqrt(sum([(r - mean)**2 for r in f1s])/len(f1s))
    print (mean,std)


def build_model(X, y, model_type, params):
    return model_type(**params).fit(X, y)


def evaluate(X, y, model):
    predictions = model.predict(X)
    return (f1_score(predictions, y), precision_score(predictions, y), recall_score(predictions, y))


def multi_evaluate(X, y, model):
    predictions = model.predict(X)
    return (f1_score(predictions, y, average="macro"), precision_score(predictions, y, average="macro"), recall_score(predictions, y, average="macro"))


def tune(data, model_type, params, percent=.8):
    classifier = model_type(**params)
    data_x = [d[:-1] for d in data]
    data_y = [d[-1] for d in data]

    train_x = data_x[:int(len(data_x)*percent)]
    train_y = data_y[:int(len(data_x)*percent)]
    test_x = data_x[int(len(data_x)*percent):]
    test_y = data_y[int(len(data_x)*percent):]

    classifier.fit(train_x, train_y)
    return evaluate(test_x, test_y, classifier)

