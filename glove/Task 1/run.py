#This file trains the model using the given glove embeddings(as path variable). It then saves test accuracy for specified number of epochs as a pickle file accuracy.pickle in same directory.  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchsummary import summary
import pickle
import pandas as pd
from sklearn import preprocessing
import numpy as np
import math
import sys

dataframe_path = sys.argv[1]
glove_path = sys.argv[2]

num_of_dims = 300
learning_rate = 0.001
EPOCHS = 1000

dataframe = pickle.load(open(dataframe_path, 'rb'))
glove_dict = pickle.load(open(glove_path, 'rb'))

def average_embedding(text):
    #This function takes as input the text sequence and then averages the embeddings of all words in it. Words out of dictionary are considered as zero vector embedding.
    average_embedding = np.zeros(num_of_dims)
    for word in text.split():
        if word in glove_dict.keys():
            average_embedding += glove_dict[word]
    return average_embedding/len(text.split())

def isnan(arr):
    if math.isnan(arr[0]):
        return 1
    else:
        return 0

dataframe['embedding'] = dataframe['text'].apply(average_embedding)
dataframe.drop(['text', 'comp_name', 'sub_sector', 'f_path'], axis=1, inplace=True)
dataframe['new'] = dataframe['embedding'].apply(isnan)
dataframe = dataframe[dataframe['new'] == 0]
dataframe =  dataframe[dataframe['sector'] !='na']

le = preprocessing.LabelEncoder()
le.fit(dataframe['sector'])
dataframe['sector'] = le.transform(dataframe['sector'])
num_of_classes = len(le.classes_)
id_label_dict = {id:label for id, label in zip(range(num_of_classes), le.classes_)}

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.X = np.ndarray((len(self.dataframe), num_of_dims), dtype = float)
        for i in range(len(self.dataframe)):
            self.X[i] = self.dataframe['embedding'].iloc[i]
        self.y = self.dataframe['sector'].values
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

train_dataframe, test_dataframe = train_test_split(dataframe, train_size = 0.8, random_state = 42)
train_dataset = CustomDataset(train_dataframe)
test_dataset = CustomDataset(test_dataframe)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(num_of_dims, 100)
        self.fc2 = nn.Linear(100, num_of_classes)
    def forward(self, x):
        #print(x.shape)
        x = F.leaky_relu(self.fc1(x))
        #print(x.shape)
        x = F.log_softmax(self.fc2(x), dim=-1)
        #print(x.shape)
        return x

net = NN()
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def train():
    accuracy = []
    for epoch in range(EPOCHS):
        batch_loss = 0
        for data in train_loader:
            inputs, labels = data[0], data[1]
            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        accuracy.append(test())
        if(not epoch%5):
            print('[Epoch %d] loss: %0.3f' % (epoch+1, batch_loss/len(train_loader)))
            print("Accuracy on test data is: %0.3f " % test())
    return accuracy

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0], data[1]
            outputs = net(inputs.float())
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100*correct/total

accuracy = train()

pickle.dump(accuracy, open('accuracy.pickle', 'wb'))