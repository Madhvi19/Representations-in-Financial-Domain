#!/usr/bin/env python
# coding: utf-8

# In[218]:


from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pickle
import numpy as np
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity as cs
import pandas as pd


# In[20]:


####
# Loading glove embeddings from pickle file glove_new.pickle and writing into embedding map and a text file which
# can be used to gensim model
####
file = open("glove_new.pickle",'rb')
embedding_map = pickle.load(file)

outfile = open("glove_new.txt",'w')

for i in list(embedding_map.keys()):
    s = ""
    word = i
    s = s+str(word)+" "
    for l in embedding_map[word]:
        s+=str(l)
        s+=" "
    s = s.strip()+"\n"
    outfile.write(s)
outfile.close()
embedding_map.clear()


# In[3]:


#####
## Vocab size of pretrained glove embeddings from stanford.
#####
with open("/Users/skosgi/Downloads/nlp/Project/glove_vocab", "rb") as vocabfile:
    vocab = pickle.load(vocabfile)
print(len(vocab))


# In[98]:


##########
##Converting glove embeddings to numpy matrix where each row contains embedding of a word.
##Adding words to "word to id" and "id to word" maps
##########
w2id = {}
id2w = {}
w = np.zeros((len(embedding_map.keys()),300))

for i,word in enumerate(embedding_map.keys()):
    w2id[word] = i
    id2w[i] = word
    w[i] = embedding_map[word]


# In[99]:


######
##Applying PCA to reduce the dimension of the embedding from 300D to 2D.
######

pca = PCA(n_components=2)
k = pca.fit_transform(w)


# In[155]:


######
##API used for returning count number of similar vectors of word for the given embeddings.
######

def similar_vectors(w,word,count):
    
    word_vec = w[w2id[word]]
    similarity = {}
    for i in range(w.shape[0]):
        new_vec = w[i]
        dot = np.dot(word_vec,new_vec)
        magnitude = np.linalg.norm(new_vec)*np.linalg.norm(word_vec)
        cosine_similarity = dot/magnitude
        similarity[i] = cosine_similarity
     
    similarity = sorted(similarity.items(),reverse=True,key = lambda similarity: (similarity[1],similarity[0]))
    k =0
    sim_vector = []
    print("------- Similar words for {}-------------".format(word))
    for i,score in similarity:
        if k==count:
            break
        sim_vector.append(i)    
        print(id2w[i])
        k+=1
    return sim_vector    


# In[227]:


similar_vectors(w,"profit",20)


# In[168]:


def cosine_similarity(word1,word2):
    word1_vec = w[w2id[word1]]
    word2_vec = w[w2id[word2]]
    dot = np.dot(word1_vec,word2_vec)
    magnitude = np.linalg.norm(word1_vec)*np.linalg.norm(word2_vec)
    cosine_sim = dot/magnitude
    return cosine_sim


# In[212]:


cosine_similarity('debt','unsecured')


# In[228]:


w_list = ['profit','positive','improved','debt','borrowings','unsecured']

scores = np.zeros((len(w_list),len(w_list)))
for i in range(len(w_list)):
    for j in range(i,len(w_list)):
        scores[i][j] = cosine_similarity(w_list[i],w_list[j])
        scores[j][i] = scores[i][j]
df = pd.DataFrame(data=scores,    # values
        index=w_list,    # 1st column as index
        columns=w_list)
print(df)


# In[94]:


#####
## Gives 2D embeddings of the words similar to a given word 
#####
def sim_embeddings(word,count):
    sim_vectors = similar_vectors(w,word,count)
    x = [k[i][0] for i in sim_vectors]
    y = [k[i][1] for i in sim_vectors]
    return x,y


# In[159]:


#####
##API used to scatter the points on 2D space
#####

def scatter_embeddings(words,count):
    colors = cm.rainbow(np.linspace(0, 1, len(words)))
    XY = []
    for i in range(len(words)):
        x,y = sim_embeddings(words[i],count)
        xy = plt.scatter(x,y,color = colors[i])
        XY.append(xy)    
    
    plt.legend((XY),(words),loc="upper left",ncol=3,
           fontsize=8)
    plt.xlabel("W1")
    plt.ylabel("W2")
    plt.title("Word vectors using Glove embeddings")


# In[160]:


#####
###Words used for plotting on 2D space
#####

words = ['bankruptcy','paid','tax','litigation','profits','debt','amazon','microsoft','astrazeneca']
#words = ['google','search','engine','astrazeneca','aceto','vaccine']

scatter_embeddings(words,20)


# In[11]:


######
##Loading gensim models to load pretrained and trained embeddings to appreciate the difference between them.
######
from gensim.models import Word2Vec,KeyedVectors
import nltk
from gensim.scripts.glove2word2vec import glove2word2vec


# In[21]:


###### 
model = glove2word2vec(glove_input_file='glove_new.txt',word2vec_output_file="glove_vectors_10k.txt")


# In[22]:


###### Gensim Model for trained embeddings on 10k corpus
model = KeyedVectors.load_word2vec_format('glove_vectors_10k.txt',binary=False)


# In[91]:


###### Most similar words of a given word from trained embeddings
model.most_similar("sec")


# In[29]:


######Gensim Model for pretrained embeddings on wiki and book corpus
model1 = KeyedVectors.load_word2vec_format('/Users/skosgi/Downloads/nlp/project/gensim_glove_vectors.txt',binary=False)


# In[92]:


###### Most similar words of a given word from pretrained embeddings
model1.most_similar("sec")


# In[ ]:




