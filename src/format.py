from src.variables import *
from typing import List
import pandas as pd

def main():
    '''
    Update: This module fetches the data from e-commerce.xlsx and outputs it to csv
    '''
    df = excel_to_df(f'{data_folder}/e-commerce.xlsx')
    print(df)
    df_to_csv(df, data_folder)

def excel_to_df(file_path: str = 'data') -> bool:
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f'Error parsing excel to dataframe {e}')
        return False

def df_to_excel(df: pd.DataFrame, data_folder: str = 'data') -> bool:
    try:
        df.to_excel(f'{data_folder}/e-commerce.xlsx')
        return True
    except Exception as e:
        print(f'Error parsing dataframe to excel {e}')
        return False

def df_to_csv(df: pd.DataFrame, data_folder: str = 'data') -> bool:
    try:
        df.to_csv(f'{data_folder}/e-commerce.csv')
        return True
    except Exception as e:
        print(f'Error parsing dataframe to csv {e}')
        return False

##############################
# Deprecated methods
##############################

def list_to_dataframe(list: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(list, columns=['sentiment', 'comment'])

    df = df[['comment', 'sentiment']] # Put the class/category (sentiment) last

    df.loc[df['sentiment'] == 'neg', 'sentiment'] = -1
    df.loc[df['sentiment'] == 'pos', 'sentiment'] = 1

    # df = df.sort_values(by='sentiment', ascending=False, ignore_index=True) # Comment to make it unordered
    df = df.sample(frac=1) # Comment if you want the original values from the txt file
    
    return df
    
def txt_to_list(filepath: str = 'data/original.txt')-> List[List[str]]:

    def split_sentiment_review(string: str):
        splitted = string.split(maxsplit=1)

        if splitted is None or len(splitted) == 0:
            return None
        
        if splitted[0] != 'pos' and splitted[0] != 'neg':
            return None
        
        return splitted

    with open(filepath, 'r', encoding='utf-8') as file:
        corpus = []

        for document in file:
            document = document.strip()
            
            # Check before split operation
            if document == '':
                continue

            splitted = split_sentiment_review(document)
            
            if splitted is None:
                continue
            
            corpus.append(splitted)
        
        return corpus

if __name__ == '__main__':
    main()