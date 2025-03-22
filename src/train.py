from src.variables import *
from src.vectorizer import Vectorizer, BOW, TFIDF
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    X_train, X_test, y_train, y_test = get_splitted_train_test()
    
    vectorizer = get_vectorizer('bow')
    X_vectorized = vectorizer.vectorize(X_train)
    print(X_vectorized)

def get_vectorizer(string: str) -> Vectorizer:
    if string == 'bow':
        return BOW()
    if string == 'TFIDF':
        return TFIDF()

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