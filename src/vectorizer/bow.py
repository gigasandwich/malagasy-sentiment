from sklearn.feature_extraction.text import CountVectorizer

def vectorize(X_train):
    count_vec = CountVectorizer()
    X_train_vectorized = count_vec.fit_transform(X_train)
    return X_train_vectorized