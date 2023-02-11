from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataloading import data_split, Glove_embedding, ClsDataset, collate_fn, make_dataloader
from module import ESIM, train, plot
import torch

learning_rate = 0.001
embedding_dim = 50
len_hidden = 50
num_epoch = 2
batch_size = 1000
train_iter, val_iter, glove_embedding = make_dataloader(batch_size=batch_size)
model = ESIM(embedding_dim, len_hidden, glove_embedding.len_words, longest=glove_embedding.longest,
             weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
train_loss, val_loss, train_acc, val_acc = train(model, train_iter, val_iter, learning_rate, num_epoch)
plot(train_loss, val_loss, train_acc, val_acc, num_epoch)