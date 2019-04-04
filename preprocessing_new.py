#!/usr/bin/env python
# coding: utf-8

import re
import string
import nltk
from nltk.corpus import stopwords
import os

path3 = f"./Data/train.tsv"
count = 0

for i in range(0, 69):
    path = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/txt/record-{i}.txt"
    path2 = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/concept/record-{i}.con"

    if(os.path.exists(path)):
        count += 1
        print(count)
        new_data = []
        with open(path, 'r') as data:
            count3 = 0
            for lines in data.readlines():
                count3 += 1
                if count3 > 8:
                    for words in lines.split():
                        words = words.lower()
                        if re.match("^[.,a-z0-9->_]*$", words):
                            new_data.append(words)
                    new_data.append('XXXXXX')

            ner_dict = dict()
            with open(path2, 'r') as ner:
                for line in ner.readlines():
                    line = line.lower()
                    matches=re.findall(r'\"(.+?)\"',line)
                    ner_dict[matches[0]] = matches[1]
            
            # stop_words = set(stopwords.words('english'))
            stop_words = {}
            # new_data = [w for w in new_data if not w in stop_words and len(w)>2]
            new_data = [w for w in new_data if w not in stop_words and len(w)>2]
            ners = dict()
            for keys in ner_dict: 
                keys = keys.lower()
                words = [w for w in keys.split() if not w in stop_words and len(w)>2 and w.isalpha()]
                word = ' '.join(map(str, words))
                if len(words)>0:
                    ners[word] = ner_dict[keys]

            for word in ners:
                if len(word.split()) > 1:
                    words = word.split()
                    indices = []
                    for i in range(len(new_data)):
                        flag = True
                        for j in range(len(words)):
                            if(new_data[i+j] == words[j]):
                                pass
                            else:
                                flag = False
                                break
                        if(flag):
                            indices.append(i)
                        for ind in indices:
                            for total in range(len(words)):
                                if len(re.split(r'\t+', new_data[ind+total]))==1:
                                    if total==0:
                                        new_data[ind+total] = f"{new_data[ind+total]}\tB-{ners[word]}"
                                    else:
                                        new_data[ind+total] = f"{new_data[ind+total]}\tI-{ners[word]}"

            for word in ners:
                if word is not 'XXXXXX':
                    if len(word.split())==1:
                        for i in range(len(new_data)):
                            if new_data[i]==word:
                                if len(re.split(r'\t+', new_data[i]))==1:
                                    new_data[i] = f"{new_data[i]}\tB-{ners[word]}"

            for i in range(len(new_data)):
                if len(re.split(r'\t+', new_data[i]))==1:
                    if new_data[i] == 'XXXXXX':
                        pass
                    else:
                        new_data[i] = f"{new_data[i]}\tO"

            with open(path3, 'a+') as outf:
                count2 = 0
                for word in new_data:
                    count2 += 1
                    if word == 'XXXXXX' and count2%8 == 0:
                        outf.write('\n')
                    elif word == 'XXXXXX' and count2%8 != 0:
                        pass
                    else:
                        outf.write(word)
                        outf.write('\n')
                # print("Len of document is : ", len(new_data))
