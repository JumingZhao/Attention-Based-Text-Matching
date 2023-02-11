import numpy
import random
from matplotlib import pyplot as plt
import re
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
def data_split(data, val_rate=0.2):
    train = []
    val = []
    i = 0
    for datum in data:
        i += 1
        if random.random() > val_rate:
            train.append(datum)
        else:
            val.append(datum)
    return train, val
class Glove_embedding():
    def __init__(self, data, trained_dict, val_rate=0.2):
        self.dict_words = {}
        self.trained_dict = trained_dict
        self.len_words = 0
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.train, self.val = data_split(self.data, val_rate=val_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]
        self.val_y = [self.type_dict[term[2]] for term in self.val]
        self.train_s1_matrix = []
        self.val_s1_matrix = []
        self.train_s2_matrix = []
        self.val_s2_matrix = []
        self.longest = 0
        self.embedding = [] # 抽取出用到的（预训练模型的）单词

    def get_words(self):
        self.embedding.append([0]*50) # 先加padding的词向量
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.lower()
                words = re.findall(pattern, s)
                for word in words:
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)
                        if word in self.trained_dict:
                            self.embedding.append(self.trained_dict[word])
                        else:
                            self.embedding.append([0] * 50)
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for term in self.train:
            s = term[0]
            s = s.lower()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s1_matrix.append(item)
            s = term[1]
            s = s.lower()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s2_matrix.append(item)
        for term in self.val:
            s = term[0]
            s = s.lower()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.val_s1_matrix.append(item)
            s = term[1]
            s = s.lower()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.val_s2_matrix.append(item)
        self.len_words+=1


class ClsDataset(Dataset):

    def __init__(self, sentence1, sentence2, relation):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.relation = relation

    def __getitem__(self, item):
        return self.sentence1[item], self.sentence2[item], self.relation[item]

    def __len__(self):
        return len(self.relation)


def collate_fn(batch_data):
    sents1, sents2, labels = zip(*batch_data)
    sentences1 = [torch.LongTensor(sent) for sent in sents1]
    padded_sents1 = pad_sequence(sentences1, batch_first=True, padding_value=0)
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    padded_sents2 = pad_sequence(sentences2, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2), torch.LongTensor(labels)


def make_dataloader(batch_size=1024):
    with open('snli_1.0/snli_1.0_train.txt', 'r') as f:
        temp = f.readlines()
    with open('glove.6B.50d.txt', 'rb') as f:
        lines = f.readlines()

    trained_dict = {}
    n = len(lines)
    for i in range(n):
        line = lines[i].split()
        trained_dict[line[0].decode("utf-8").lower()] = [float(line[j]) for j in range(1, 51)]

    glove_embedding = Glove_embedding(data=temp[1:], trained_dict=trained_dict)
    glove_embedding.get_words()
    glove_embedding.get_id()

    train_dataset = ClsDataset(glove_embedding.train_s1_matrix, glove_embedding.train_s2_matrix,
                               glove_embedding.train_y)
    val_dataset = ClsDataset(glove_embedding.val_s1_matrix, glove_embedding.val_s2_matrix,
                             glove_embedding.val_y)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                collate_fn=collate_fn)

    return train_dataloader, val_dataloader, glove_embedding
