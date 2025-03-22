from gensim.models import Word2Vec

def vectorize(X_train):
    w2v_model = Word2Vec(X_train, vectorsize=100)

    def get_vector(X: str):
        return sum(w2v_model.wv[word] for word in X if word in w2v_model.wv) / len(X)

    X_train_vectorized = [get_vector(comment) for comment in X_train]
    return X_train_vectorized