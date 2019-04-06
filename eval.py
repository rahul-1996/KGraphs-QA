from data_load import HParams
from ner_model import Net
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert import BertModel
import parameters
import numpy as np 
import torch

config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)

def build_model(config, state_dict, hp):
    model = Net(config, vocab_len = len(hp.VOCAB), bert_state_dict=None)
    _ = model.load_state_dict(torch.load(state_dict, map_location='cpu'))
    _ = model.to('cpu')  # inference 
    return model 


i2b2_model = build_model(config, parameters.I2b2_WEIGHTS, HParams('i2b2'))

# Process Query 
def process_query(query, hp, model):
    s = query
    split_s = ["[CLS]"] + s.split()+["[SEP]"]
    x = [] # list of ids
    is_heads = [] # list. 1: the token is the first piece of a word

    for w in split_s:
        tokens = hp.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
        xx = hp.tokenizer.convert_tokens_to_ids(tokens)
        is_head = [1] + [0]*(len(tokens) - 1)
        x.extend(xx)
        is_heads.extend(is_head)

    x = torch.LongTensor(x).unsqueeze(dim=0)

    # Process query 
    model.eval()
    hp = HParams('i2b2')
    hidden = model.init_eval_hidden(hp.batch_size)
    _, _, y_pred = model(x, hidden)  # just a dummy y value
    preds = y_pred[0].cpu().numpy()[np.array(is_heads) == 1]  # Get prediction where head is 1 

    # convert to real tags and remove <SEP> and <CLS>  tokens labels 
    preds = [hp.idx2tag[i] for i in preds][1:-1]
    final_output = []
    for word, label in zip(s.split(), preds):
        final_output.append([word, label])
    return final_output



def get_ner(query):
    hp = HParams('i2b2')
    print("i2b2 -> ", query)
    out = process_query(query=query, hp=hp, model=i2b2_model)
    result = []
    ners = []
    for op in out:
        if op[1] == 'O':
            result.append(op[0])
        elif op[1] == 'B-problem':
            result.append('problem')
            ners.append(op[0])
        elif op[1] == 'I-problem':
            ners[-1] = f"{ners[-1]} {op[0]}"
        elif op[1] == 'B-test':
            result.append('test')
            ners.append(op[0])
        elif op[1] == 'I-test':
            ners[-1] = f"{ners[-1]} {op[0]}"
        elif op[1] == 'B-treatment':
            result.append('treatment') 
            ners.append(op[0])
        elif op[1] == 'I-treatment':
            ners[-1] = f"{ners[-1]} {op[0]}"
    return result, ners

if __name__ == '__main__':
    query = input()
    result = get_i2b2(query)
    print(result)

