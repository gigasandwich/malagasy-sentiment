filepath = 'data/original.txt'

def main():
    corpus = format_to_list(filepath)
    print(corpus)

def format_to_list(filepath):
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

def split_sentiment_review(string: str):
    splitted = string.split(maxsplit=1)

    if splitted is None or len(splitted) == 0:
        return None
    
    if splitted[0] != 'pos' and splitted[0] != 'neg':
        return None
    
    return splitted

if __name__ == '__main__':
    main()