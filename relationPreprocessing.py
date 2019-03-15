
# coding: utf-8

# In[ ]:


#Imports
import sys


# In[48]:


#Input path declarations
entity_path = "data_new.tsv"
relation_path = "../concept_assertion_relation_training_data/beth/rel"
clinical_notes_path = "../concept_assertion_relation_training_data/beth/txt"


# In[78]:


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


# In[180]:


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


# In[184]:


#Read the relations file
import os
files = os.listdir(relation_path)
count = 0
for file in files:
    out=[]
    path = relation_path + "/" + file
    print(path)
    recordPath = clinical_notes_path + "/" + file.split(".")[0] + ".txt"
    lines = open(path).read().strip().split('\n')
    for line in lines:
        print(line)
        if len(line.strip()) == 0 :
            break
        entity1 = line.split("\"")[1]
        relation = line.split("\"")[3]
        entity2 = line.split("\"")[5]
        line = int(line.split("\"")[2].lstrip().split(":")[0])
        print(entity1, entity2, relation, line)
        print(get_line_from_file(recordPath ,line))
        textLine = get_line_from_file(recordPath ,line)
        entityTypes = getEntityTypes(relation)
        textLine=textLine.replace(entity1, entityTypes[0])
        textLine=textLine.replace(entity2, entityTypes[1])
        out.append(textLine + "\t" + relation)
    with open("relationsTrain.tsv", 'a+') as outf:
                outf.write(file.split(".")[0] + ":\n")
                for line in out:
                    outf.write(line)
                    outf.write('\n')
                outf.write('\n\n\n\n')
        
        



# In[178]:




