from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize(X_train):
    tfidf = TfidfVectorizer()
    X_train_vectorized = tfidf.fit_transform(X_train)
    return X_train_vectorized