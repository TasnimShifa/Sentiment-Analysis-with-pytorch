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
def Test(model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total_data = 0
    n=0
    with torch.no_grad():
        for batch in test_loader:
            n = n+1
            text, label = batch.text.to(device), (batch.label-1).to(device)  #label: 1-5
            output = model(text)
            loss = criterion(output, label)
            test_loss += loss.item()
            total_data += label.size(0)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= n
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, total_data,
        100. * correct / (total_data)))
    
    return (test_loss, 1.0*correct / (total_data))


