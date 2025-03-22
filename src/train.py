from src.variables import *
from src.vectorizer import Vectorizer, BOW, TFIDF, WORD_EMBEDDINGS
from src.classification_model import Model, NaiveBayesModel, LogisticRegressionModel, RandomForestModel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Literal

def main():
    '''
    Here, we:
    1) Choose a vectorizer with the get_vectorizer method
    2) Vectorize X_train with the X_train_vectorized variable
    3) Choose a model with the get_classification_model method
    '''
    X_train, X_test, y_train, y_test = get_splitted_train_test()
    
    # 1
    vectorizer = get_vectorizer('BOW') # Just change the argument if you want to test another one
    
    # 2
    X_train_vectorized = vectorizer.vectorize(X_train, fit=True)
    
    # 3
    model = get_classification_method('LogisticRegressionModel')
    model.fit(X_train_vectorized, y_train)
    
    # When using the model for predictions, we must not learn a new vocabulary
    # Instead, we apply the same transformation learned from X_train, so fit=False
    X_test_vectorized = vectorizer.vectorize(X_test, fit=False)
    
    y_pred = model.predict(X_test_vectorized)
    # print(f'Accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}')

def get_classification_method(string: Literal['NaiveBayesModel', 'LogisticRegressionModel', 'RandomForestModel']) -> Model:
    models = {
        'NaiveBayesModel': NaiveBayesModel(),
        'LogisticRegressionModel': LogisticRegressionModel(),
        'RandomForestModel': RandomForestModel()
    }
    return models.get(string, NaiveBayesModel())

def get_vectorizer(string: Literal['BOW', 'TFIDF', 'WORD_EMBEDDINGS']) -> Vectorizer:
    vectorizers = {
        'BOW': BOW(),
        'TFIDF': TFIDF(),
        'WORD_EMBEDDINGS': WORD_EMBEDDINGS()
    }
    return vectorizers.get(string, BOW())

def get_splitted_train_test():
    df = load_data(f'{datafolder}/english.csv')
    df = df.head(100) # Uncomment to use all data

    X = df['comment'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    main()