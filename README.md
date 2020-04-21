# Sentence Level Sentiment Multiclass Classification with Pytorch


* Dataset: Stanford Sentiment Treebank
* Embedding: Glove (glove.6B.300d.txt)
* Model: LSTM


# Prerequisite
* torch version : 1.0.0
* torchtext version : 0.5.0

# Run
Gitclone or download the project and simply use run.sh bash file 
```
$ bash run.sh

```
Or, run the MTT.py file with necessary parameter settings. For example,
```
$ python MTT.py --lr=0.001 --itr=3 --dropout=0.5 --hidden_dim=256
```
Or, check main.ipynb for jupyter notebook environment.
