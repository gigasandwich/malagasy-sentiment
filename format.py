filePath = 'data/original.txt'

with open(filePath, 'r') as file:
    count = 0
    for line in file:
        line = line.strip()
        print(line)
        
        count += 1
        if count == 20:
            break