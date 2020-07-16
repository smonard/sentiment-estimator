import pickle
import numpy
import spacy
import keras
import joblib
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from keras.models import load_model    

class BasicSentimentClassifier:
    def __init__(self, vectorizer_path, selector_path):
        self.__nlp = spacy.load('es_core_news_sm')
        self.__vectorizer_path, self.__selector_path = vectorizer_path, selector_path
        self.__vectorizer, self.__selector = None, None
        try:
            with open(vectorizer_path, 'rb') as file:
                self.__vectorizer = pickle.load(file)
            with open(selector_path, 'rb') as file:
                self.__selector = pickle.load(file)
        except Exception as e:
            print("{}. Consider model training, using fit(X,y)".format(e))
            
    def __fit_features(self, featureset):
        lemmatized = self.__lemmatize(featureset)
        self.__vectorizer = TfidfVectorizer() if self.__vectorizer == None else  self.__vectorizer
        vectorization = self.__vectorizer.fit_transform(lemmatized)
        self.__selector = VarianceThreshold() if self.__selector == None else self.__selector
        return self.__selector.fit_transform(vectorization.toarray())
    
    def _dumpmodel(self):
        with open(self.__vectorizer_path, 'wb') as file:
            pickle.dump(self.__vectorizer, file)
        with open(self.__selector_path, 'wb') as file:
            pickle.dump(self.__selector, file)
    
    def __lemmatize(self, texts):
        return [' '.join([tk.lemma_.lower() if tk.pos_ != 'PROPN' else 'name' for tk in self.__nlp(text)]) for text in texts]

    def __extract_features(self, texts):
        lemmatized = self.__lemmatize(texts)
        vectorized = self.__vectorizer.transform(lemmatized).toarray()
        return numpy.array(self.__selector.transform(vectorized)).astype('float64')

    def predict(self, text):
        features = self.__extract_features([text])
        return self._predict(features)
    
    def fit(self, X, y, reset=True):
        print("\n Extracting features")
        X = self.__fit_features(X)
        y = numpy.array(y).astype('float64')
        print("\n Training model")
        return self._fit(X, y, reset)
    
    def score(self, X, y):
        print("\n Evaluating model")
        X = self.__extract_features(X)
        return self._score(X, y)
        
    @staticmethod
    def newinstance(modeltype = 'nn', context_path = '../data/'):
        return NeuralNetworkSentimentClassifier(context_path=context_path) if modeltype == 'nn' else NaiveBayesSentimentClassifier(context_path=context_path)
    
    
class NeuralNetworkSentimentClassifier(BasicSentimentClassifier):
    def __init__(self, vectorizer_path = 'vectorizer.obj', selector_path = 'selector.obj', model_path = 'model_clf_nn.obj', context_path='/'):
        self.__model_path = context_path + model_path
        try:
            self._model_clf = load_model(self.__model_path)
        except:
            self._model_clf = None
        super().__init__(context_path + vectorizer_path, context_path + selector_path)

    def __parse_nnresult(self, rs):
        plain_result = numpy.argmax(rs)
        case = {0: lambda x: 1.0 - x[0], 1:lambda x: 1.0 + (rs[2] - rs[0]), 2: lambda x: 1.0 + x[2]}.get(plain_result)
        return case(rs)
    
    def __create_model(self, x_train, y_train, output_length):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(len(x_train[0]),)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(output_length, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])
        return model

    def _predict(self, features):
        prediction = self._model_clf.predict(features)
        return self.__parse_nnresult(prediction[0])
    
    def _fit(self, X, y, reset):
        output = len(set(y))
        y = keras.utils.to_categorical(y, output)
        self._model_clf = self.__create_model(X, y, output) if reset == True or self._model_clf == None else self._model_clf
        history = self._model_clf.fit(X, y,
                       batch_size = 128,
                       epochs = 20,
                       verbose = 0)
        self._model_clf.save(self.__model_path)
        self._dumpmodel()
        return history
        
    def _score(self, X, y):
        output = len(set(y))
        y = keras.utils.to_categorical(y, output)
        return self._model_clf.evaluate(X, y)[1]

class NaiveBayesSentimentClassifier(BasicSentimentClassifier):
    def __init__(self, vectorizer_path = 'vectorizernb.obj', selector_path = 'selectornb.obj', model_path = 'model_clf_bayes.obj', context_path='./'):
        self.__model_path = context_path + model_path
        try:
            self._model_clf = joblib.load(self.__model_path)
        except:
            self._model_clf = None
        super().__init__(context_path + vectorizer_path, context_path + selector_path)

    def _predict(self, features):
        prediction = self._model_clf.predict(features)
        return prediction[0]
    
    def _fit(self, X, y, reset):
        self._model_clf = ComplementNB() if reset == True or self._model_clf == None else self._model_clf
        train_result = self._model_clf.fit(X, y)
        joblib.dump(self._model_clf, self.__model_path)
        self._dumpmodel()
        return train_result
    
    def _score(self, X, y):
        return self._model_clf.score(X, y)
