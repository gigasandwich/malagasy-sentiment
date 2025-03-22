from src.variables import *
from typing import List
import pandas as pd

def main():
    '''
    This module formats and exports the original file 'data/original.txt' to match the common csv standarts (comment: str, sentiment: int)
    '''
    corpus = txt_to_list(f'{data_folder}/original.txt')
    df = list_to_dataframe(corpus)
    print(df)
    df_to_csv(df)

def df_to_csv(df: pd.DataFrame, data_folder: str = 'data') -> bool:
    try:
        df.to_csv(f'{data_folder}/english.csv', index=False)
        return True
    except:
        return False

def list_to_dataframe(list: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(list, columns=['sentiment', 'comment'])

    df = df[['comment', 'sentiment']] # Put the class/category (sentiment) last

    df.loc[df['sentiment'] == 'neg', 'sentiment'] = -1
    df.loc[df['sentiment'] == 'pos', 'sentiment'] = 1

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