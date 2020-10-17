from glove import Corpus, Glove
import os
import re
import sys
path = sys.argv[1]
freq = int(sys.argv[2])
window = int(sys.argv[3])
dim = int(sys.argv[4])
lr = float(sys.argv[5])
epochs = int(sys.argv[6])

def train(path, freq, window, dim, lr, epochs):
    lines = []
    dic = {}
    print("Start of train method")
    try:
        for f in os.listdir(path):
            text = open(path + '/' + f, 'r').read()
            text = re.sub('\n', ' ', text)
            text = text.split()
            for word in text:
                if word in dic.keys():
                    dic[word] += 1
                else:
                    dic[word] = 1
        print("Created Dictionary for frequencies of words.")
        for f in os.listdir(path):
            text = open(path + '/' + f, 'r').read()
            text = re.sub('\n', ' ', text)
            text = text.split()
            text = [word for word in text if dic[word]>freq]
            lines.append(text)
        print("Converted preprocessed text data in input format of array of array of words.")
        corpus = Corpus()
        corpus.fit(lines, window=window)
        glove = Glove(no_components=dim, learning_rate=lr)
        glove.fit(corpus.matrix, epochs=epochs, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        glove.save('glove.model')
        print("Saved the trained model to glove.model.")
    except:
        print("Error occured in training glove model")

def main():
    train(path, freq, window, dim, lr, epochs)

if __name__ == "__main__":
    main()


