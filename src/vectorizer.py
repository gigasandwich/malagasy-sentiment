from abc import ABC, abstractmethod

##############################
# Original class
##############################

class Vectorizer(ABC):
    '''
    We will call these vectorizers to parse the string to numbers for the model
    '''
    @abstractmethod
    def vectorize(self, X_train: str):
        pass

##############################
# Extended classes
##############################

class BOW(Vectorizer):
    def vectorize(self, X_train):
        from sklearn.feature_extraction.text import CountVectorizer

        count_vec = CountVectorizer()
        X_train_vectorized = count_vec.fit_transform(X_train)
        return X_train_vectorized

class TFIDF(Vectorizer):
    def vectorize(self, X_train):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        tfidf = TfidfVectorizer()
        X_train_vectorized = tfidf.fit_transform(X_train)
        return X_train_vectorized

class WORD_EMBEDDINGS(Vectorizer):

    def vectorize(self, X_train):
        from gensim.models import Word2Vec
        
        w2v_model = Word2Vec(X_train, vectorsize=100)

        def get_vector(X: str):
            return sum(w2v_model.wv[word] for word in X if word in w2v_model.wv) / len(X)

        X_train_vectorized = [get_vector(comment) for comment in X_train]
        return X_train_vectorized