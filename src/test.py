from src.variables import *
import joblib

def main():
    model_path = f'{trained_models_folder}/naivebayesmodel-tfidf.pkl'
    model, vectorizer = load_model(model_path)

    X_new = [
        'This is the finest product I bought',
        'I really hate the experience, very bad service!',
        'It s okay',
        'Absolutely amazing! I will buy again',
        'Worst product I bought ever'
    ]
    X_new_vectorized = vectorizer.transform(X_new)
    y_new_predicted = model.predict(X_new_vectorized)

    for comment, sentiment in zip(X_new, y_new_predicted):
        print(f'{comment}: {sentiment}')

def load_model(model_path):
    try:
        loaded_data = joblib.load(model_path)
        return loaded_data['model'], loaded_data['vectorizer']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

if __name__ == '__main__':
    main()