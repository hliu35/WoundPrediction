import os

class generatePairs():
    for entry in os.listdir('data/'):
        if(entry.endswith("A8-1-L.png")):
            print(entry)


generatePairs()