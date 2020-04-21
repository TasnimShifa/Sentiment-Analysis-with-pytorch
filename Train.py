import argparse
import torch
torch.cuda.set_device(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchvision import datasets, transforms
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram, FastText
from RNN import Model
import easydict 


#################################
#functions
def Train( model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    cnt = 0;
    for batch in train_loader:
        text, label = batch.text.to(device), (batch.label-1).to(device)  #label: 1-5
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))


