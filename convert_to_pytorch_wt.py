#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import re
import torch
import numpy as np


# In[2]:


tf_path = 'weights/biobert_model.ckpt'


# In[3]:


init_vars = tf.train.list_variables(tf_path)


# In[4]:


excluded = ['BERTAdam','_power','global_step']
init_vars = list(filter(lambda x:all([True if e not in x[0] else False for e in excluded]),init_vars))


# In[5]:


init_vars


# In[25]:


names = []
arrays = []
for name, shape in init_vars:
    print("Loading TF weight {} with shape {}".format(name, shape))
    array = tf.train.load_variable(tf_path, name)
    names.append(name)
    arrays.append(array)


# In[7]:


from pytorch_pretrained_bert  import BertConfig, BertForPreTraining


# In[9]:


# Initialise PyTorch model
config = BertConfig.from_json_file('weights/bert_config.json')
print("Building PyTorch model from configuration: {}".format(str(config)))
model = BertForPreTraining(config)


# In[10]:



for name, array in zip(names, arrays):
    name = name.split('/')
    # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
    # which are not required for using pretrained model
    if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
        print("Skipping {}".format("/".join(name)))
        continue
    pointer = model
    for m_name in name:
        if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
            l = re.split(r'_(\d+)', m_name)
        else:
            l = [m_name]
        if l[0] == 'kernel' or l[0] == 'gamma':
            pointer = getattr(pointer, 'weight')
        elif l[0] == 'output_bias' or l[0] == 'beta':
            pointer = getattr(pointer, 'bias')
        elif l[0] == 'output_weights':
            pointer = getattr(pointer, 'weight')
        else:
            pointer = getattr(pointer, l[0])
        if len(l) >= 2:
            num = int(l[1])
            pointer = pointer[num]
    if m_name[-11:] == '_embeddings':
        pointer = getattr(pointer, 'weight')
    elif m_name == 'kernel':
        array = np.transpose(array)
    try:
        assert pointer.shape == array.shape
    except AssertionError as e:
        e.args += (pointer.shape, array.shape)
        raise
    print("Initialize PyTorch weight {}".format(name))
    pointer.data = torch.from_numpy(array)

# Save pytorch-model


# In[11]:


print("Save PyTorch model to {}".format('weights/'))
torch.save(model.state_dict(),'weights/pytorch_weight')


# In[ ]:





# In[13]:


import os
import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from pytorch_pretrained_bert import BertConfig, BertForPreTraining

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    config_path = os.path.abspath(bert_config_file)
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {} with config at {}".format(tf_path, config_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


# In[ ]:




