
# coding: utf-8

# In[1]:


#Imports
import sys


# In[2]:


#Input path declarations
entity_path = "data_new.tsv"
relation_path = "../concept_assertion_relation_training_data/beth/rel"
clinical_notes_path = "../concept_assertion_relation_training_data/beth/txt"


# In[3]:


#Read the entity tagged file -> Has actual sentences and entities
instances = open(entity_path).read().strip().split('\n\n\n\n\n')
sents = {}
tags_li = {}
print(len(instances))
for entry in instances:
#     print(entry.splitlines())
    record = entry.splitlines()[0].split(":")[0]
    words = [line.split()[0] for line in entry.splitlines()]
    tags = ([line.split()[-1] for line in entry.splitlines()])
    words = words[1:]
    tags=tags[1:]
    sents[record]= words
    tags_li[record] =tags
print(sents, tags_li)
print(sents.keys())
print(len(sents) )


# In[4]:


def getEntityTypes(r):   
    if(r == 'TrCP'):
        #Treatment caused Problem
        entity1 = "Treatment"
        entity2 = "Problem"
    elif(r=='TeCP'):
        #Test to detect Problem (outcome not known)
        entity1 = "Test"
        entity2 = "Problem"
    elif(r == 'TrWP'):
        #Treatment made problem worse
        entity1 = "Treatment"
        entity2 = "Problem"
    elif(r == "TeRP" ):
        #A test revealed a problem (outcome known)
        entity1 = "Test"
        entity2 = "Problem"
    elif(r == "PIP"):
        #One medical problem (symptom) reveals or vauses another medical problem (disease)
        entity1= "Problem1"
        entity2= "Problem2"
    elif(r == "TrAP"):
        #Treatment given for a problem (Outcome not known)
        entity1 = "Treatment"
        entity2 = "Problem"
    elif(r == "TrNAP"):
        #Treatment not adminstered/ stopped
        entity1= "Treatment"
        entity2 = "Problem"
    elif(r == "TrIP"):
        #Treatment imporves problem (Outcome Known)
        entity1 = "Treatment"
        entity2 = "Problem"
    return entity1 , entity2

def get_line_from_file(recordPath ,line):
    lines = open(recordPath).read().strip().split('\n')
    return lines[line-1]
    


# In[34]:


#Read the relations file
import os
import pdb
files = os.listdir(relation_path)
count = 0
for file in files:
    out=[]
    path = relation_path + "/" + file
    print(path)
    recordPath = clinical_notes_path + "/" + file.split(".")[0] + ".txt"
    lines = open(path).read().strip().split('\n')
    line_numbers=[]
    for line in lines:
        print(line)
        if len(line.strip()) == 0 :
            break
        entity1 = line.split("\"")[1]
        relation = line.split("\"")[3]
        entity2 = line.split("\"")[5]
        line_num = int(line.split("\"")[2].lstrip().split(":")[0])   
        textLine = get_line_from_file(recordPath ,line_num)
        entityTypes = getEntityTypes(relation)
        textLine=textLine.lower().replace(entity1.lower(), entityTypes[0].lower())
        textLine=textLine.lower().replace(entity2.lower(), entityTypes[1].lower())
        line_numbers.append(line_num)     
        out.append(textLine + "\t" + relation)    
    lines = open(recordPath).read().strip().split('\n')
    print(len(lines))
    line_numbers = list(set(line_numbers))
    print(line_numbers)
    for i in range(len(lines)):
        
        print(get_line_from_file(recordPath , i) , (len(get_line_from_file(recordPath , i).strip().split())))
        if(((i+1) in line_numbers)  or (len(get_line_from_file(recordPath , i).strip().split())<5)):
            lines.remove(get_line_from_file(recordPath , i))

        print(i)
    print(len(lines))
    lines= lines[:min(50, len(lines))]
    print(len(lines))

    for x in lines:
        out.append(x + "\t" + "None")
    
    with open("relationsTrain.tsv", 'a+') as outf:
#                 outf.write(file.split(".")[0] + ":\n")
                for line in out:
                    outf.write(line)
                    outf.write('\n')
#                 outf.write('\n\n\n\n')
        
        



# In[178]:


get_line_from_file(recordPath , i)

