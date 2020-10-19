import gensim
import logging
import re
import os
import sys
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO) 

#sample command to run the file in terminal. Change the directory path and hyperparameters accordingly.
#python word2vec.py D:\College\Study\IRE\Project\data\clean10k\home\ubuntu\data\clean 100 5 5 4 1 0 5 

path = sys.argv[1]
size = int(sys.argv[2]) 
window = int(sys.argv[3]) 
min_count = int(sys.argv[4]) 
workers = int(sys.argv[5]) 
sg = int(sys.argv[6]) 
hs = int(sys.argv[7]) 
negative = int(sys.argv[8])

class Files(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            yield re.sub('\n', ' ', open(os.path.join(self.dirname, fname)).read()).split()

def train():
    files = Files(path)
    model = gensim.models.Word2Vec(files, size=size, window=window, min_count=min_count, workers=workers, sg=sg, hs=hs, negative=negative)
    model.save("word2vec.model")

def main():
    train()

if __name__ == "__main__":
    main()

