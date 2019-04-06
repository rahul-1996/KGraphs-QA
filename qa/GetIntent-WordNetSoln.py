
# coding: utf-8

# In[10]:


import pandas as pd
d = pd.read_excel("QA.xlsx", sheetname='Question-Intent')


# In[11]:


from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
def check_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
 
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
 
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
 
    score, count = 0.0, 0
    #print(synsets1)
    #print(synsets2)
       # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        best_score = list([synset.path_similarity(ss) for ss in synsets2])
        best_score= list(filter(lambda a: a != None, best_score))
        if(best_score==[]):
            best_score =0
        else:
            best_score = max(best_score)
        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1
 
    # Average the values
    if (count!= 0):
        score /= count
    return score


# In[12]:


def sortThird(val): 
    return val[2]  
  
def get_Intent(q):
    all=[]
    for x in list(d["Question"]):
       # print(x  , "||" , list(d[d["Question"] == x]["Intent"])[0], "||", check_similarity(q, x))
        l=[]
        l.append(x)
        l.append(list(d[d["Question"] == x]["Intent"])[0])
        l.append(check_similarity(q, x))
        all.append(l)
    all.sort(key = sortThird, reverse=True)
    return all[0][1]


# In[13]:
if __name__ == '__main__':
    query = "Does probleme also cause problem"
    intent = get_Intent(query)
    print("The intent is" , intent)

