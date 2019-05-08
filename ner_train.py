import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from ner_model import Net
from data_load import NerDataset, pad_ner, HParams
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig
import parameters
from collections import OrderedDict 
import json
from torch.autograd import Variable
from sklearn.metrics import f1_score
from functools import partial
import pickle
from sklearn.metrics import precision_recall_fscore_support
# device = 'cpu'
device = 'cuda'

model_state_dict = torch.load('./weights/save_file')

hp = HParams('i2b2')
clip = 5

def train(model, iterator, optimizer, criterion):

    model.train()
    model = model.to(device)
    hidden = model.init_hidden(hp.batch_size)
    
    for i, batch in enumerate(iterator):
        if(i < 30):
            words, x, is_heads, tags, y, seqlens = batch
            _y = y # for monitoring
            hidden = tuple([each.data for each in hidden])

            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            logits, hidden, _ = model(x, hidden) # logits: (N, T, VOCAB), y: (N, T)

            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)

            loss = criterion(logits, y)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            if i%6==0: # monitoring
                print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f):
    
    model.eval()
    model.to(device)
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    hidden = model.init_hidden(hp.batch_size)
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            if i<10:
                words, x, is_heads, tags, y, seqlens = batch
                x = x.to(device)
                y = y.to(device)
                _, _, y_hat = model(x, hidden)  # y_hat: (N, T)
                Words.extend(words)
                Is_heads.extend(is_heads)
                Tags.extend(tags)
                Y.extend(y.cpu().numpy().tolist())
                Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open(f, 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [hp.idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([hp.tag2idx[line.split()[1]] for line in open(f, 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([hp.tag2idx[line.split()[2]] for line in open(f, 'r').read().splitlines() if len(line) > 0])

    score = precision_recall_fscore_support(y_true,y_pred,average='weighted')

    precision, recall, f1 = score[0], score[1], score[2]
    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={score[0]}\n")
        fout.write(f"recall={score[1]}\n")
        fout.write(f"f1={score[2]}\n")

    os.remove(f)

    print("precision=%.2f"%score[0])
    print("recall=%.2f"%score[1])
    print("f1=%.2f"%score[2])
    return score[0], score[1], score[2]

if __name__=="__main__":

    train_dataset = NerDataset("Data/train.tsv", 'i2b2')  
    eval_dataset = NerDataset("Data/test.tsv", 'i2b2')
    
    # Define model
    config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
    model = Net(config = config, bert_state_dict = model_state_dict, vocab_len = len(hp.VOCAB), device=hp.device)
    
    # 'bc5cdr': ('<PAD>', 'B-Chemical', 'O', 'B-Disease' , 'I-Disease', 'I-Chemical'),

    class_sample_count = [10, 1, 20, 3, 4] # dataset has 10 class-1 samples, 1 class-2 samples, etc.
    weights = 1 / torch.Tensor(class_sample_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, hp.batch_size)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 collate_fn=pad_ner)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 collate_fn=pad_ner)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train_on_gpu = True
    if(train_on_gpu):
        model.cuda()

    for epoch in range(1, 31):
        train(model, train_iter, optimizer, criterion)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        fname = os.path.join('checkpoints', str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname)
        torch.save(model.state_dict(), f"{fname}.pt")
