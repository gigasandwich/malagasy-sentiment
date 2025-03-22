from src.variables import *
from typing import List
import pandas as pd

def main():
    corpus = txt_to_list(filepath)
    df = list_to_dataframe(corpus)
    print(df)

    df_to_csv(df)

def df_to_csv(df: pd.DataFrame, datafolder: str = 'data'):
    df.to_csv(f'{datafolder}/english.csv', index=False)

def list_to_dataframe(list: List[str]):
    df = pd.DataFrame(list, columns=['sentiment', 'comment'])

    df = df[['comment', 'sentiment']] # Put the class/category (sentiment) last

    df.loc[df['sentiment'] == 'neg', 'sentiment'] = -1
    df.loc[df['sentiment'] == 'pos', 'sentiment'] = 1

    return df
    
def txt_to_list(filepath: str = 'data/original.txt'):

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