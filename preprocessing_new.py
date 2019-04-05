#!/usr/bin/env python
# coding: utf-8

import re
import string
import nltk
from nltk.corpus import stopwords
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


path3 = f"./Data/train.tsv"
count = 0

new_data = []

for i in range(0, 69):
    path = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/txt/record-{i}.txt"
    path2 = f"/Users/rahulsm/Downloads/concept_assertion_relation_training_data/beth/concept/record-{i}.con"

    if(os.path.exists(path)):
        count += 1
        print(count)
        with open(path, 'r') as data:
            count3 = 0
            for lines in data.readlines():
                for words in lines.split():
                    words = words.lower()
                    if re.match("^[.,a-z0-9->_]*$", words):
                        new_data.append(words)

            ner_dict = dict()
            with open(path2, 'r') as ner:
                for line in ner.readlines():
                    line = line.lower()
                    matches=re.findall(r'\"(.+?)\"',line)
                    ner_dict[matches[0]] = matches[1]
            
            stop_words = set(stopwords.words('english'))
            # stop_words = {}
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
                                        new_data[ind+total] = f"{new_data[ind+total]}\tB-{ners[word]}"

            for word in ners:
                if len(word.split())==1:
                    for i in range(len(new_data)):
                        if new_data[i]==word:
                            if len(re.split(r'\t+', new_data[i]))==1:
                                new_data[i] = f"{new_data[i]}\tB-{ners[word]}"

            for i in range(len(new_data)):
                if len(re.split(r'\t+', new_data[i]))==1:
                        new_data[i] = f"{new_data[i]}\tO"



count_O = 0
count_test = 0
count_treatment = 0
count_problem = 0

nonO_data = []

for word in new_data:
    words = word.split('\t')
    if words[1] == 'O':
        count_O += 1
    elif words[1] == 'B-treatment':
        count_treatment +=1
        nonO_data.append(word)
    elif words[1] == 'B-test':
        count_test += 1
        nonO_data.append(word)
    elif words[1] == 'B-problem':
        count_problem += 1
        nonO_data.append(word)

print(f"O = {count_O}, treatment = {count_treatment}, test = {count_test}, problem = {count_problem}")



temp_treatment = []
temp_test = []
temp_problem = []

for i, word in enumerate(new_data):
    words = word.split('\t')
    if words[1] == 'B-treatment':
        temp_treatment.append(word)
        index = i+1
        while (index < len(new_data)) and (new_data[index].split('\t')[1] == 'I-treatment'):
            temp_treatment.append(new_data[index])
            index = index + 1

    elif words[1] == 'B-test':
        temp_test.append(word)
        index = i+1
        while (index < len(new_data)) and (new_data[index].split('\t')[1] == 'I-test'):
            temp_test.append(new_data[index])
            index = index + 1


    elif words[1] == 'B-problem':
        temp_problem.append(word)
        index = i+1
        while (index < len(new_data)) and (new_data[index].split('\t')[1] == 'I-problem'):
            temp_problem.append(new_data[index])
            index = index + 1

print(f"Len of treatment is : {len(temp_treatment)}, problem: {len(temp_problem)}, test: {len(temp_test)}")


import random


temp = []

while count_treatment < 0.7*count_O: 
    index = random.randint(0, len(temp_treatment))
    if (index < len(temp_treatment)) and (temp_treatment[index].split('\t')[1] == 'B-treatment'):
         temp.append(temp_treatment[index])
         index = index + 1
         count_treatment += 1
         while (index < len(temp_treatment)) and (temp_treatment[index].split('\t')[1] == 'I-treatment'):
             temp.append(temp_treatment[index])
             index = index + 1

print(f"O = {count_O}, treatment = {count_treatment}, test = {count_test}, problem = {count_problem}")  
new_data.extend(temp)

temp = []

while count_test < 0.7*count_O: 
    index = random.randint(0, len(temp_test))
    if (index < len(temp_test)) and (temp_test[index].split('\t')[1] == 'B-test'):
         temp.append(temp_test[index])
         index = index + 1
         count_test += 1
         while (index < len(temp_test)) and (temp_test[index].split('\t')[1] == 'I-test'):
             temp.append(temp_test[index])
             index = index + 1

print(f"O = {count_O}, treatment = {count_treatment}, test = {count_test}, problem = {count_problem}")  
new_data.extend(temp)

temp = []

while count_problem < 0.7*count_O: 
    index = random.randint(0, len(temp_problem))
    if (index < len(temp_problem)) and (temp_problem[index].split('\t')[1] == 'B-problem'):
         temp.append(temp_problem[index])
         index = index + 1
         count_problem += 1
         while (index < len(temp_problem)) and (temp_problem[index].split('\t')[1] == 'I-problem'):
             temp.append(temp_problem[index])
             index = index + 1

print(f"O = {count_O}, treatment = {count_treatment}, test = {count_test}, problem = {count_problem}")  
new_data.extend(temp)

print(f"length of new_data is : {len(new_data)}")


final_data = []

count = 0

while (count+100) < len(new_data):
        tmp = new_data[count: count+100]
        final_data.append(tmp)
        count = count + 100

print(f"Len of final data is : {len(final_data)}")
random.shuffle(final_data)

fin_data = []
for arr in final_data:
    fin_data.extend(arr)

print(f"Len of fin data is : {len(fin_data)}")


with open(path3, 'a+') as outf:
    count2 = 0
    count3 = 0
    # while count3 <= 100000000:
    for word in fin_data:
        count2 += 1
        if count2%100 == 0:
            outf.write('\n')
            count3 += 1
        outf.write(word)
        outf.write('\n')
    print("Count2 is ", count2)
    print("Count3 is ", count3)
