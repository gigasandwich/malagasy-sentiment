from src.variables import *
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    df = load_data(f'{datafolder}/english.csv')

    X = df['comment'].values
    y = df['sentiment'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    main()