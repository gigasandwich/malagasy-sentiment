from src.variables import *
from src.vectorizer import Vectorizer, BOW, TFIDF, WORD_EMBEDDINGS
from src.classification_model import Model, Naive_Bayes, LogisticRegression, RandomForest

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal

def main():
    '''
    Here, we: \n
    1) Choose a vectorizer with the get_vectorizer method \n
    2) Vectorize X_train with the X_vectorized variable
    3) Choose a model with the get_classification_model method
    '''
    X_train, X_test, y_train, y_test = get_splitted_train_test()
    
    # 1
    vectorizer = get_vectorizer('BOW') # Just change the argument if you want to test another one
    
    # 2
    X_vectorized = vectorizer.vectorize(X_train)
    
    # 3
    model = get_classification_method('Naive_Bayes')
    model.fit(X_vectorized, y_train)
    
    # Evaluate the model
    y_pred = model.pre    

def get_classification_method(string: Literal['Naive_Bayes', 'LogisticRegression', 'RandomForest']) -> Model:
    if string == 'Naive_Bayes':
        return Naive_Bayes()
    if string == 'LogisticRegression':
        return LogisticRegression()
    if string == 'RandomForest':
        return RandomForest()

def get_vectorizer(string: Literal['BOW', 'TFIDF', 'WORD_EMBEDDINGS']) -> Vectorizer:
    if string == 'BOW':
        return BOW()
    if string == 'TFIDF':
        return TFIDF()
    if string == 'WORD_EMBEDDINGS':
        return WORD_EMBEDDINGS()

def get_splitted_train_test():
    df = load_data(f'{datafolder}/english.csv')

    X = df['comment'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    main()