from glove import Corpus, Glove
import os
import re

path = ""

def train(path):
    lines = []
    dic = {}
    for f in os.listdir(path):
        text = open(path + '/' + f, 'r').read()
        re.sub('\n', '', text)
        text = text.split()
        for word in text:
            if word in dic.keys():
                dic[word] += 1
            else:
                dic[word] = 1
    for f in os.listdir(path):
        text = open(path + '/'+f, 'r').read()
        re.sub('\n', '', text)
        text = text.split()
        text = [word for word in text if dic[word]>5]
        lines.append(text)
    corpus = Corpus()
    corpus.fit(lines, window=10)
    glove = Glove(no_components=100, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=20, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove.model')

def main():
    train(path)

if __name__ == "__main__":
    main()


