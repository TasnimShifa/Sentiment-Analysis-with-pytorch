import torch
import torch.nn.functional as F


# (seqLen, batchSize, unitNums)
class Model(torch.nn.Module):
    def __init__(self,embeddingDim, embeddingNum, unitNums,dropout = 0.,dim1=150,  dim2=10, labels = 5):
        super().__init__()
        
        self.d = dropout
        
        self.embedding = torch.nn.Embedding(embeddingNum, embeddingDim)
        self.embeddingNum = embeddingNum
        self.embeddingDim = embeddingDim
        self.unitNums = unitNums
        self.dim1 = dim1
        self.dropout = torch.nn.Dropout(dropout)
        self.dim2 = dim2
        self.labels = labels
        self.W1 = torch.nn.Parameter(torch.randn(unitNums,dim1))
        self.W2 = torch.nn.Parameter(torch.randn(dim1,dim2))
        self.fcInput = unitNums*dim2
        self.fc = torch.nn.Linear(unitNums*dim2, labels)
        self.model = torch.nn.LSTM(embeddingDim, unitNums) 
        
    def forward(self,x):
        with torch.no_grad():
            embedded_x = self.embedding(x)
        output, _ = self.model(embedded_x) #shape(output): (seqLen, batchSize, numUnits)
        output = output.permute(1,0,2) #(batchSize,seqLen,numUnits)
        output_T = output.permute(0,2,1) #(batchSize,numUnits,seqLen)
        #using attention method
        attention_T =  torch.softmax(torch.einsum('ijk,kl->ijl', torch.tanh(torch.einsum('ijk,kl->ijl', (output, self.W1))), self.W2), 1) # (batchSize, seqLen, dim2)
        result_T = torch.matmul(output_T, attention_T) #(batchSize,numUnits,dim2)
        result = result_T.permute(0,2,1) #(batchSize,dim2,numUnits)
        
        if self.d > 0:
        
            input = self.dropout(result.reshape((-1, self.fcInput)))
            logits = self.fc(input) # batchSize, labels
            return logits
        
        else :
            logits = self.fc(result.reshape((-1, self.fcInput)))
            return logits
