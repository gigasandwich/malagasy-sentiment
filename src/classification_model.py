from abc import ABC, abstractmethod
from typing import List
##############################
# Original class
##############################

class Model(ABC):
    @abstractmethod
    def fit(self, X_train: List[str], y_train: List[str]):
        pass

    def predict(self, X_test):
        return self.clf.predict(X_test)

##############################
# Extended classes
##############################

class NaiveBayesModel(Model):
    '''
    For simple text data
    '''

    def fit(self, X_train, y_train):
        from sklearn.naive_bayes import MultinomialNB

        self.clf = MultinomialNB()
        self.clf.fit(X_train, y_train)

class LogisticRegressionModel(Model):
    '''
    For small datasets
    '''
    def fit(self, X_train, y_train):
        from sklearn.linear_model import LogisticRegression

        self.clf = LogisticRegression()
        self.clf.fit(X_train, y_train)

class RandomForestModel(Model):
    '''
    Handles non-linearity
    '''
    def fit(self, X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier

        self.clf = RandomForestClassifier()
        self.clf.fit(X_train, y_train)