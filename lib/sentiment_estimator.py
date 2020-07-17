from sentiment_classifier import SentimentClassifier

class SentimentEstimator:
    def __init__(self, context_path):
        self.__nn_estimator = SentimentClassifier.newinstance(context_path=context_path)
        self.__nb_estimator = SentimentClassifier.newinstance('nb', context_path=context_path)
    
    def predict(self, text):
        sentiment = (self.__nn_estimator.predict(text) + self.__nb_estimator.predict(text)) / 2.0
        return sentiment
