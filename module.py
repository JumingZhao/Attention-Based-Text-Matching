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
from dataloading import data_split, Glove_embedding, ClsDataset, collate_fn, make_dataloader

class Input_Encoding(nn.Module):
    def __init__(self, embedding_dim, len_hidden, len_words, longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest=longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, embedding_dim))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=embedding_dim, _weight=x).cuda()
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=embedding_dim, _weight=weight).cuda()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()

    def forward(self, x):
        x = torch.LongTensor(x).cuda()
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class LocalInference(nn.Module):
    def __init__(self):
        """
        p代表：premise
        h代表：hypothesis
        """

        super(LocalInference, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).cuda()
        self.softmax_2 = nn.Softmax(dim=2).cuda()
        """
        点积打分模型，直接相乘然后softmax就是注意力分布，总的分布矩阵[batch,seq_len_p,seq_len_h]
        这个矩阵横着看每一行代表以premise的一个词向量为query查询的注意力分布                         
        纵向看每一列代表以hypothesis的一个词向量维query查询的注意力的分布

        """

    def forward(self, p, h):
        e = torch.matmul(p, h.transpose(1, 2)).cuda()

        p_ = self.softmax_2(e)
        p_ = p_.bmm(h)
        h_ = self.softmax_1(e)
        h_ = h_.transpose(1, 2).bmm(p)

        maskp = torch.cat([p, p_, p - p_, p * p_], dim=-1)
        maskh = torch.cat([h, h_, h - h_, h * h_], dim=-1)

        return maskp, maskh

class Inference_Composition(nn.Module):
    def __init__(self, embedding_dim, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):

        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, embedding_dim).cuda()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).cuda()
        self.dropout = nn.Dropout(drop_out).cuda()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)

        return output

class Prediction(nn.Module):
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num)).cuda()

    def forward(self, p, h):

        vp_avg = p.sum(1) / p.shape[1] #平均池
        vp_max = p.max(1)[0]       #最大池

        vh_avg = h.sum(1) / h.shape[1]
        vh_max = h.max(1)[0]

        out_put = torch.cat((vp_avg, vp_max, vh_avg, vh_max), dim=-1)

        return self.mlp(out_put)

class ESIM(nn.Module):
    def __init__(self, embedding_dim, len_hidden, len_words, longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words = len_words
        self.longest = longest
        self.input_encoding = Input_Encoding(embedding_dim, len_hidden, len_words, longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        self.localInference = LocalInference()
        self.inference_composition = Inference_Composition(embedding_dim, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction = Prediction(len_hidden*8, len_hidden, type_num=type_num, drop_out=drop_out)

    def forward(self,p,h):
        p_bar = self.input_encoding(p)
        h_bar = self.input_encoding(h)

        maskp, maskh = self.localInference(p_bar, h_bar)

        v_p = self.inference_composition(maskp)
        v_h = self.inference_composition(maskh)

        out_put = self.prediction(v_p,v_h)

        return out_put


def train(model, train_iter, val_iter, learning_rate, num_epoch):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record = []
    val_loss_record = []
    train_acc_record = []
    val_acc_record = []

    for epoch in range(num_epoch):
        torch.cuda.empty_cache()
        model.train()
        for i, batch in enumerate(train_iter):
            torch.cuda.empty_cache()
            x1, x2, y = batch
            pred = model(x1, x2).cuda()
            optimizer.zero_grad()
            y = y.cuda()
            loss = loss_fun(pred, y).cuda()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            train_acc = []
            val_acc = []
            train_loss = 0
            val_loss = 0
            for i, batch in enumerate(train_iter):
                torch.cuda.empty_cache()
                x1, x2, y = batch
                y = y.cuda()
                pred = model(x1, x2).cuda()
                loss = loss_fun(pred, y).cuda()
                train_loss += loss.item()
                _, y_pre = torch.max(pred, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                train_acc.append(acc)

            for i, batch in enumerate(val_iter):
                torch.cuda.empty_cache()
                x1, x2, y = batch
                y = y.cuda()
                pred = model(x1, x2).cuda()
                loss = loss_fun(pred, y).cuda()
                val_loss += loss.item()
                _, y_pre = torch.max(pred, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                val_acc.append(acc)

        trains_acc = sum(train_acc) / len(train_acc)
        vals_acc = sum(val_acc) / len(val_acc)

        train_loss_record.append(train_loss / len(train_acc))
        val_loss_record.append(val_loss / len(val_acc))
        train_acc_record.append(trains_acc.cpu())
        val_acc_record.append(vals_acc.cpu())
        print("---------- Epoch", epoch + 1, "----------")
        print("Train loss:", train_loss / len(train_acc))
        print("test loss:", val_loss / len(val_acc))
        print("Train accuracy:", trains_acc)
        print("test accuracy:", vals_acc)

    return train_loss_record, val_loss_record, train_acc_record, val_acc_record


def plot(train_loss, val_loss, train_acc, val_acc, num_epoch):
    x = list(range(1, num_epoch + 1))
    plt.subplot(1, 2, 1)
    plt.plot(x, train_loss, label='train', color='red')

    plt.legend(fontsize=10)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(x, train_acc, label='train', color='red')
    plt.plot(x, val_acc, label='test', color='blue')
    plt.legend(fontsize=10)
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(12, 4, forward=True)
    plt.show()
