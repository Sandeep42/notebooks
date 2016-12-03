
# coding: utf-8

# In[1]:

import spacy
from collections import defaultdict, Counter


# In[3]:

from spacy.en import English
engine = English()


# In[4]:

with open('trump_speech_54.txt','r') as f:
    speech = f.read()


# In[ ]:

#print(speech)


# In[5]:

parsedSpeech = engine(speech)


# In[6]:

type(parsedSpeech)


# In[7]:

parsedSpeech.sents


# In[13]:

pos_counts = defaultdict(Counter)


# ## Lemmatization

# In[8]:

for i, token in enumerate(parsedSpeech):
    print("original:", token.orth_)
    print("lowercased:", token.lower_)
    print("lemma:", token.lemma_)
    print("----------------------------------------")
    if i > 25:
        break


# ## POS tagging

# In[9]:

s = 'As you have heard, it was just announced yesterday that the FBI is reopening their investigation into the criminal and illegal conduct of Hillary Clinton.'


# In[10]:

parsedSpeech2 = engine(s)


# In[11]:

for token in parsedSpeech2:
    print(token.orth_, token.tag_)
    print('----------------------')


# In[14]:

for token in parsedSpeech:
    pos_counts[token.pos][token.orth] += 1

for pos_id, counts in sorted(pos_counts.items()):
    pos = parsedSpeech.vocab.strings[pos_id]
    for orth_id, count in counts.most_common(5):
        print(pos, count, parsedSpeech.vocab.strings[orth_id])
    print('-----------')


# ## Named entities

# In[15]:

with open('trump_speech_54.txt','r') as f:
    speech = f.read()


# In[16]:

parsedSpeech = engine(speech)


# In[17]:

ents = list(parsedSpeech.ents)


# In[18]:

for entity in ents:
    print(entity.label_, ' '.join(t.orth_ for t in entity))
    print('-------------------------------------')


# In[ ]:



