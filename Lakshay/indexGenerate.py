# Creating index 

from string import ascii_lowercase
import itertools

def createIndex(n):
    def iter():
        for size in itertools.count(1):
            for s in itertools.product(ascii_lowercase, repeat = size):
                yield "".join(s)
                
    text = []
    for s in itertools.islice(iter(),n):
        #print s
        #text[s] = s
        text.append(s)
        
    #text.to_csv("alphabet_index.csv")
    #print(text)
    return text