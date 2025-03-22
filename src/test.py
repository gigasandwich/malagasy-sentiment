from src.variables import *
import joblib

def main():
    model_path = f'{trained_models_folder}/logisticregressionmodel-bow.pkl'
    model = joblib.load(model_path)
        

if __name__ == '__main__':
    main()