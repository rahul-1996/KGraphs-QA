import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from ner_model import Net
from relation_model import RelNet
from data_load import NerDataset, pad, HParams, device, RelationDataset
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig
import parameters
from collections import OrderedDict 
import json
from torch.autograd import Variable
import pdb; 

# prepare biobert dict
# tmp_d = torch.load(parameters.BERT_CONFIG_FILE, map_location=lambda storage, location: 'cpu')

tmp_d = {
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 2048,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 28996
}

state_dict = OrderedDict()
for i in list(tmp_d.keys())[:199]:
    x = i
    if i.find('bert') > -1:
        x = '.'.join(i.split('.')[1:])
    state_dict[x] = tmp_d[i]


#defining hidden state
# hidden = torch.rand(2*2, hp.batch_size, hp.hidden_size)
# nn.init.xavier_normal_(hidden)
# c_0 = torch.rand(2*2, hp.batch_size, hp.hidden_size)
# nn.init.xavier_normal_(c_0)

clip = 5

def train(model, iterator, optimizer, criterion):
    model.train()
    hidden = model.init_hidden(hp.batch_size)
    for i, batch in enumerate(iterator):
        # pdb.set_trace()
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        hidden = tuple([each.data for each in hidden])

        optimizer.zero_grad()
        logits, hidden = model(x, hidden) # logits: (N, T, VOCAB), y: (N, T)

        # logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        # y = y.view(-1)  # (N*T,)

        loss = criterion(logits.squeeze(), y.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # if i==0:
        #     print("=====sanity check======")
        #     print("x:", x.cpu().numpy()[0])
        #     print("words:", words[0])
        #     print("tokens:", hp.tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0]))
        #     print("y:", _y.cpu().numpy()[0])
        #     print("is_heads:", is_heads[0])
        #     print("tags:", tags[0])
        #     print("seqlen:", seqlens[0])


        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
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

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open(f, "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove(f)

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

if __name__=="__main__":

    hp = HParams('i2b2')

    # train_dataset = NerDataset("Data/train.tsv", 'i2b2')  
    # eval_dataset = NerDataset("Data/test.tsv", 'i2b2')

    # print('reached hereeeeee')
    # # Define model
    # config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
    # model = Net(config = config, bert_state_dict = state_dict, vocab_len = len(hp.VOCAB), device=hp.device)
    # # model.cuda()
    # model.train()

    # train_iter = data.DataLoader(dataset=train_dataset,
    #                              batch_size=hp.batch_size,
    #                              shuffle=True,
    #                              num_workers=4,
    #                              collate_fn=pad)
    # eval_iter = data.DataLoader(dataset=eval_dataset,
    #                              batch_size=hp.batch_size,
    #                              shuffle=False,
    #                              num_workers=4,
    #                              collate_fn=pad)

    # optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)

    # #updating hidden

    # for epoch in range(1, 31):
    #     train(model, train_iter, optimizer, criterion)
    #     print(f"=========eval at epoch={epoch}=========")
    #     if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
    #     fname = os.path.join('checkpoints', str(epoch))
    #     precision, recall, f1 = eval(model, eval_iter, fname)
    #     torch.save(model.state_dict(), f"{fname}.pt")

    #Adding relations model 

    hp = HParams('relations')
    relations_train_dataset = RelationDataset("Data/formatted/relationsTrain.tsv", 'relations')  
    relations_eval_dataset = RelationDataset("Data/formatted/relationsTest.tsv", 'relations')

    print('reached hereeeeee')
    
    # Define model
    config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
    model = RelNet(config = config, bert_state_dict = state_dict, vocab_len = len(hp.VOCAB), device=hp.device)
    # model.cuda()
    model.train()

    train_iter = data.DataLoader(dataset=relations_train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=relations_eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    # optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    #updating hidden

    for epoch in range(1, 31):
        train(model, train_iter, optimizer, criterion)
        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
        fname = os.path.join('checkpoints', str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname)
        torch.save(model.state_dict(), f"{fname}.pt")