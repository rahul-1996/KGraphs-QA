
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import pdb
from numpy import array
from relation_model import RelNet
from data_load import NerDataset, pad_rel, HParams,RelationDataset
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig
import parameters
from collections import OrderedDict 
import json
from torch.autograd import Variable

from sklearn.metrics import precision_recall_fscore_support

model_state_dict = torch.load('./weights/save_file')

state_dict = torch.load('weights/save_file')

clip = 5
train_on_gpu=torch.cuda.is_available()
if(train_on_gpu):
    device = 'cuda'

def train(model, iterator, optimizer, criterion):
    model.train()
    hidden = model.init_hidden(hp.batch_size)
    for i, batch in enumerate(iterator):
        # pdb.set_trace()
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        hidden = tuple([each.data for each in hidden])

        if(train_on_gpu):
            x,y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        logits, hidden, _ = model(x,hidden) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if i%20==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f):
    model.eval()
    hidden = model.init_hidden(hp.batch_size)
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            if(train_on_gpu):
                x,y = x.cuda(), y.cuda()
            logits, _, y_hat = model(x,hidden)  # y_hat: (N, T)
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y2 = y.view(-1)  # (N*T,)

            loss = criterion(logits, y2)
            Words.extend(words)
            # Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())
    #print(Y_hat)
    Preds = [hp.idx2tag[hat] for hat in Y_hat]
    ## gets results and save
    with open(f, 'w') as fout:
        for t,p in zip( Tags, Preds):
            # y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            # preds = [hp.idx2tag[hat] for hat in y_hat]
            # print(preds)
            # assert len(preds)==len(words.split())==len(tags.split())
            # for t, p in zip(tags, preds):
            fout.write(f"{t} {p}\n")
            # fout.write("\n")

    ## calc metric
    y_true =  np.array([hp.tag2idx[line.split()[0]] for line in open(f, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([hp.tag2idx[line.split()[1]] for line in open(f, 'r').read().splitlines() if len(line) > 0])
    num_proposed = len(Preds)
    num_correct = np.sum(array(Preds)==array(Tags))
    # num_gold = len(y_true[y_true>1])
    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    #print(f"num_gold:{num_gold}")
    try:
        accuracy = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0
    score = precision_recall_fscore_support(y_true,y_pred,average='weighted')

    precision, recall, f1 = score[0], score[1], score[2]
    # try:
    #    recall = num_correct / num_gold
    #except ZeroDivisionError:
    #    recall = 1.0

    #try:
    #    f1 = 2*precision*recall / (precision + recall)
    #except ZeroDivisionError:
    #    if precision*recall==0:
    #        f1=1.0
    #    else:
    #        f1=0
    # recall = 0.0
    # f1 = 0.0
    final = f + ".P%.2f_R%.2f_F%.2f_A%.2f" %(precision, recall, f1, accuracy)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")
        fout.write(f"accuracy={accuracy}\n")

    os.remove(f)

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    print("accuracy=%.2f"%accuracy)
    return precision, recall, f1

if __name__=="__main__":

    hp = HParams('i2b2')

    train_on_gpu=torch.cuda.is_available()
    hp = HParams('relations')
    relations_train_dataset = RelationDataset("Data/formatted/relationsTrainFinal.tsv", 'relations')  
    relations_eval_dataset = RelationDataset("Data/formatted/relationsTestFinal.tsv", 'relations')
    
    # Define model
    config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)

    model = RelNet(config = config, bert_state_dict = state_dict, vocab_len = len(hp.VOCAB), device=hp.device)
    if(train_on_gpu): 
        model.cuda()
    model.train()

    train_iter = data.DataLoader(dataset=relations_train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 collate_fn=pad_rel)
    eval_iter = data.DataLoader(dataset=relations_eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 collate_fn=pad_rel)

    # optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    #updating hidden

    for epoch in range(1, 31):
        train(model, train_iter, optimizer, criterion)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints-rel'): os.makedirs('checkpoints-rel')
        fname = os.path.join('checkpoints-rel', str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname)
        torch.save(model.state_dict(), f"{fname}.pt")
