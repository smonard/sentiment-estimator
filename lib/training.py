import json
import collections
from sentiment_classifier import SentimentClassifier

def extract_data(path='../../spanish-corpus/sentiment-analysis/full_tagged_data.json'):
    with open(path, 'r') as file:
        data = file.read()
    train_data_raw = json.loads(data)
    X = [ s[0] for s in train_data_raw]
    y = [ s[2] for s in train_data_raw]
    print(str(len(X)) + ' Examples\nCategories:')
    print(collections.Counter(y))
    return X, y

def split_data(X, y, cutpoint=0.9):
    cutpoint = int(len(X) * cutpoint)
    x_train, x_test = X[:cutpoint], X[cutpoint:]
    y_train, y_test = y[:cutpoint], y[cutpoint:]
    return (x_train, y_train), (x_test, y_test)

def fit_models(X, y):
    nn_classifier = SentimentClassifier.newinstance('nn')
    nn_classifier.fit(X, y)
    nb_classifier = SentimentClassifier.newinstance('nb')
    nb_classifier.fit(X, y)
    return nn_classifier, nb_classifier

def score_models(nn, naive, X_t, y_t):
    print("Complement Naive Bayes: {}".format(naive.score(X_t, y_t)))
    print("Fully Connected Neural Network: {}".format(nn.score(X_t, y_t)))

def train():
    X, y = extract_data()
    (x_train, y_train), (x_test, y_test) = split_data(X, y)
    nn_classifier, nb_classifier = fit_models(x_train, y_train)
    score_models(nn_classifier, nb_classifier, x_test, y_test)

train()
