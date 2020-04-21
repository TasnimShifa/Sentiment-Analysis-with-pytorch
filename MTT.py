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
from Train import Train
from Test import Test


def cut(a,n):
    return [sum(a[i:i+n])/n for i in range(0,len(a),n)]




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training on SST dataset using RNN')

    # model hyper-parameter variables
    parser.add_argument('--lr', default=0.001, metavar='lr', type=float, help='Learning rate')
    parser.add_argument('--itr', default=5, metavar='iter', type=int, help='Number of iterations')
    parser.add_argument('--dropout', default=0.5, metavar='dropout', type=float, help='Dropout Value')
    parser.add_argument('--hidden_dim', default=256, metavar='hidden_dim', type=int, help='Number hidden units')
     
    args = parser.parse_args()
    
    
    num_iter = args.itr
    lr = args.lr
    hidden_dim = args.hidden_dim
    drop_out = args.dropout

    
    #print(num_iter,lr,hidden_dim,drop_out)
    
    criterion = nn.CrossEntropyLoss()  
    criterion.cuda()
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1234)
    device = torch.device("cuda" if use_cuda else "cpu")

    TEXT = data.Field()
    LABEL = data.Field(sequential=False,dtype=torch.long)

    batch_size = 32

    train, val, test = datasets.SST.splits(
        TEXT, LABEL, fine_grained=True, train_subtrees=False)

    TEXT.build_vocab(train, vectors=torchtext.vocab.Vectors("glove.6B.300d.txt", cache='./data'), max_size=20000)
    LABEL.build_vocab(train)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size)

    batch = next(iter(train_iter)) # for batch in train_iter

    pretrained_embeddings = TEXT.vocab.vectors
    shape = pretrained_embeddings.shape

    model = Model(shape[1],shape[0],hidden_dim,drop_out)
    model.cuda()
    print('Model Summary:\n',model)
    model.embedding.weight.data.copy_(pretrained_embeddings)


    Loss_001 =[]
    Acc_001 = []
    Loss_005 =[]
    Acc_005 = []
    Loss_01 =[]
    Acc_01 = []

    
    momentum = 0.9
    weight_decay = 0.0005



    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    print("Start Training:")
    for epoch in range(num_iter):
        Train(model, device, train_iter, optimizer, epoch,criterion)
        loss, acc = Test(model, device, test_iter,criterion)
        Loss_005.append(loss)
        Acc_005.append(acc)


Loss_005 = cut(Loss_005,5)
Acc_005 = cut(Acc_005,5)



