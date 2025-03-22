from abc import ABC, abstractmethod
from typing import List
##############################
# Original class
##############################

class Model(ABC):
    @abstractmethod
    def fit(X_train: List[str], y_train: List[str]):
        pass

##############################
# Extended classes
##############################

class Naive_Bayes(Model):
    '''
    For simple text data
    '''
    def fit(X_train, y_train):
        from sklearn.naive_bayes import MultinomialNB

        clf = MultinomialNB()
        clf.fit(X_train, y_train)

class LogisticRegression(Model):
    '''
    For small datasets
    '''
    def fit(X_train, y_train):
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression()
        clf.fit(X_train, y_train)

class LogisticRegression(Model):
    '''
    Handles non-linearity
    '''
    def fit(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)