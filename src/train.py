from src.variables import *
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Literal, Union
import joblib

def main():
    '''
    Here, we:
    1) Choose a vectorizer with the get_vectorizer method
    2) Vectorize X_train with the X_train_vectorized variable
    3) Choose a model with the get_classification_model method
    '''
    X_train, X_test, y_train, y_test = get_splitted_train_test()
    
    # 1 Possible choices: ['BOW', 'TFIDF', 'WORD_EMBEDDINGS'] # TODO: Add WORD_EMBEDDINGS
    vectorizer = get_vectorizer('TFIDF') # Just change the argument if you want to test another one
    
    # 2
    X_train_vectorized = vectorizer.fit_transform(X_train) # We fit
    
    # 3 Possible choices: ['NaiveBayes', 'LogisticRegression', 'RandomForest]
    model = get_classification_method('RandomForest')
    model.fit(X_train_vectorized, y_train)
    
    # When using the model for predictions, we must not learn a new vocabulary
    # Instead, we apply the same transformation learned from X_train, so we don't fit anymore
    X_test_vectorized = vectorizer.transform(X_test)

    y_pred = model.predict(X_test_vectorized)
    print(f'Accuracy: {accuracy_score(y_true=y_test, y_pred=y_pred)}')
    print(f'Classification report:  {classification_report(y_true=y_test, y_pred=y_pred, zero_division=1)}')

    save_model(model, vectorizer)

def save_model(model, vectorizer):
    model_names = {
        'MultinomialNB': 'NaiveBayes',
        'LogisticRegression': 'LogisticRegression',
        'RandomForestClassifier': 'RandomForest',
    }
    vectorizer_names = {
        'CountVectorizer': 'BOW',
        'TfidfVectorizer': 'TFIDF',
    }

    model_name: str = model_names.get(model.__class__.__name__).lower()
    vectorizer_name: str = vectorizer_names.get(vectorizer.__class__.__name__).lower()

    model_filename = f'{model_name}-{vectorizer_name}.pkl'
    model_path = f'{trained_models_folder}/{model_filename}'
    
    try:
        joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path)
        print(f'Model saved at: {model_path}')
    except Exception as e:
        print(f"Error saving model: {e}")


def get_classification_method(string: Literal['NaiveBayes', 'LogisticRegression', 'RandomForest']) -> BaseEstimator:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        'NaiveBayes': MultinomialNB(),
        'LogisticRegression': LogisticRegression(),
        'RandomForest': RandomForestClassifier()
    }
    return models.get(string, MultinomialNB())

def get_vectorizer(string: Literal['BOW', 'TFIDF', 'WORD_EMBEDDINGS']) -> Union[BaseEstimator, TransformerMixin]:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    vectorizers = {
        'BOW': CountVectorizer(),
        'TFIDF': TfidfVectorizer(),
        # 'WORD_EMBEDDINGS': WORD_EMBEDDINGS() # TODO: implement WORD_EMBEDDINGS
    }
    return vectorizers.get(string, CountVectorizer())

def get_splitted_train_test():
    df = load_data(f'{data_folder}/e-commerce.csv')
    # df = df.head(30) # Comment to use all data

    X = df['comment-mg'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    main()