from src.variables import *
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    X_train, X_test, y_train, y_test = get_splitted_train_test()

def bow_vectorizer(X_train):
    from src.vectorizer.bow import vectorize
    return vectorize(X_train)

def tfidf_vectorizer(X_train):
    from src.vectorizer.tfidf import vectorize
    return vectorize(X_train)

def word_embeddings_vectorizer(X_train):
    from src.vectorizer.word_embeddings import vectorize
    return vectorize(X_train)

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