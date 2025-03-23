from src.variables import *
import joblib

def main():
    model_name = f'randomforest-tfidf.pkl'
    model, vectorizer = load_model(f'{trained_models_folder}/{model_name}')

    X_new = [
        'Ity no vokatra tsara indrindra novidiko',
        'Tena halako ilay izy, serivisy ratsy be!',
        'Milay izy izany',
        'Hividy hafa koa aho amin ny manaraka',
        'Vokatra ratsy indrindra novidiko hatrizay',
        'Omeko 10/10 izany, tena tsara',
        'Tsy tsara',
        'Tsara',
        'Vokatra ara-barotra tena tsara',
        'Aza mividy ity vokatra ity',
        'Nahoana ny olona no mivarotra an ity, ity no vokatra ratsy indrindra eto'
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