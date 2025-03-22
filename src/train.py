from src.variables import *
import pandas as pd

def main():
    df = load_data(f'{datafolder}/english.csv')

def load_data(filepath: str):
    df = pd.read_csv(filepath)
    return df

if __name__ == '__main__':
    main()