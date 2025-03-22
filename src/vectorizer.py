from abc import ABC, abstractmethod
import numpy as np

##############################
# Original class
##############################

class Vectorizer(ABC):
    '''
    We will call these vectorizers to parse the string to numbers for the model
    '''
    @abstractmethod
    def vectorize(self, X_train: str, fit=True):
        pass

##############################
# Extended classes
##############################

class BOW(Vectorizer):
    def __init__(self):
        from sklearn.feature_extraction.text import CountVectorizer
        self.vect = CountVectorizer()

    def vectorize(self, X_train, fit=True):
        if fit:
            return self.vect.fit_transform(X_train)
        return self.vect.transform(X_train)

class TFIDF(Vectorizer):
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vect = TfidfVectorizer()

    def vectorize(self, X_train, fit=True):
        if fit:
            return self.vect.fit_transform(X_train)
        else:
            return self.vect.transform(X_train)

class WORD_EMBEDDINGS(Vectorizer):
    # TODO: Add fit argument here too 
    def vectorize(self, X_train):
        from gensim.models import Word2Vec
        w2v_model = Word2Vec(X_train, vector_size=100)

        def get_vector(X: str):
            valid_vectors = [w2v_model.wv[word] for word in X if word in w2v_model.wv]
            return sum(valid_vectors) / len(valid_vectors) if valid_vectors else np.zeros(100)  # For no division by zero error

        X_train_vectorized = [get_vector(comment) for comment in X_train]
        return X_train_vectorized