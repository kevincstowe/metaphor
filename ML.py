from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import cross_val_score


def run_cv(X, y, model_type, params, folds=5):
    classifier = model_type(**params)
    scores = cross_val_score(classifier, X, y, cv=folds, scoring=make_scorer(f1_score))
    #scores = cross_val_score(classifier, X, y, cv=folds, scoring="f1")
    return (sum(scores)/folds)


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

