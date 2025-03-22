from src.variables import *
import joblib

def main():
    model_name = f'logisticregressionmodel-tfidf.pkl'
    model, vectorizer = load_model(f'{trained_models_folder}/{model_name}')

    X_new = [
        'This is the best product I bought',
        'I really hate the experience, very bad service!',
        'It s okay',
        'Absolutely amazing! I will buy again',
        'Worst product I bought ever',
        'I rate it 10/10, very good',
        'Not good',
        'good',
        'Very good product',
        'Do not buy this product',
        'Why do people sell this, it s the worse product here'
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