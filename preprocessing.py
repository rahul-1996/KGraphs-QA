#!/usr/bin/env python
# coding: utf-8

import re
import string
import nltk
from nltk.corpus import stopwords
import os

path3 = f"./Data/train.tsv"
count = 0

for i in range(70, 180):
    path = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/txt/record-{i}.txt"
    path2 = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/concept/record-{i}.con"

    if(os.path.exists(path)):
        count += 1
        if count!=-1:
            print(count)
            new_data = []
            with open(path, 'r') as data:
                for lines in data.readlines():
                    for words in lines.split():
                        words = words.lower()
                        if re.match("^[a-z0-9_]*$", words):
                            new_data.append(words)


            ner_dict = dict()
            with open(path2, 'r') as ner:
                for line in ner.readlines():
                    line = line.lower()
                    matches=re.findall(r'\"(.+?)\"',line)
                    ner_dict[matches[0]] = matches[1]
        

            # stop_words = set(stopwords.words('english'))
            # data = ' '.join(map(str,new_data))
            # match = re.search(r'\b(blood)\b', data)

            
            
            stop_words = set(stopwords.words('english'))
            stop_words.update(('admission','date','discharge','birth', 'sex', 'report', 'end'))
            new_data = [w for w in new_data if not w in stop_words and len(w)>2 and w.isalpha()]

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
                                    if total==len(words)-1:
                                        new_data[ind+total] = f"{new_data[ind+total]}\t{ners[word]}"
                                    else:
                                        new_data[ind+total] = f"{new_data[ind+total]}\t{ners[word]}"

            for word in ners:
                if len(word.split())==1:
                    for i in range(len(new_data)):
                        if new_data[i]==word:
                            if len(re.split(r'\t+', new_data[i]))==1:
                                new_data[i] = f"{new_data[i]}\t{ners[word]}"


            for i in range(len(new_data)):
                if len(re.split(r'\t+', new_data[i]))==1:
                    new_data[i] = f"{new_data[i]}\tO"

            # new_data = new_data[0:2047]
            with open(path3, 'a+') as outf:
                for word in new_data:
                        outf.write(word)
                        outf.write('\n')
                outf.write('\n\n\n\n')
                print("Len of document is : ", len(new_data))




