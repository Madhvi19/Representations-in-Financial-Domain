from glove import Glove
import pickle

path = r"D:\College\Study\IRE\Project\data\staticEmbeddingModels\glove\glove.model"

dic = {}

model = Glove.load(path)

for word_id in model.dictionary.keys():
    dic[word_id] = model.word_vectors[model.dictionary[word_id]]

pickle.dump(dic, open('glove.pickle', 'wb'))