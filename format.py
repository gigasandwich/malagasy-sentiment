import pandas as pd

filepath = 'data/original.txt'

def main():
    corpus = format_to_list(filepath)
    df = list_to_dataframe(corpus)
    print(df)

def list_to_dataframe(list):
    df = pd.DataFrame(list, columns=['sentiment', 'comment'])

    df = df[['comment', 'sentiment']] # Put the class/category (sentiment) last

    df.loc[df['sentiment'] == 'neg', 'sentiment'] = -1
    df.loc[df['sentiment'] == 'pos', 'sentiment'] = 1

    return df
    
def format_to_list(filepath):

    def split_sentiment_review(string: str):
        splitted = string.split(maxsplit=1)

        if splitted is None or len(splitted) == 0:
            return None
        
        if splitted[0] != 'pos' and splitted[0] != 'neg':
            return None
        
        return splitted

    with open(filepath, 'r') as file:
        count = 0 # Added a limit because the file is too large
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

            count += 1
            if count == 20:
                break
        
        return corpus

if __name__ == '__main__':
    main()