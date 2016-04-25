
# coding: utf-8

# # TME 5 TAL
# 

# In[41]:

import numpy as  np
import codecs
import matplotlib.pyplot as plt
import unicodedata
import re
from collections import Counter,defaultdict
import nltk.corpus.reader as pt
import os

import string
import pdb
import nltk
import scipy


# ## 1. Chargement des données (automatique)
# 
# 

# In[2]:

import nltk.corpus.reader as pt
path2data = u'20news-bydate/20news-bydate-train/'

rdr = pt.CategorizedPlaintextCorpusReader(path2data, '.*/[0-9]+', encoding='latin1', cat_pattern='([\w\.]+)/*')
docs = [[rdr.raw(fileids=[f]) for f in rdr.fileids(c) ] for c in rdr.categories()]


# In[ ]:




# ## 2. Chargement des données (à la main)

# In[3]:

def readAFile(nf):
    f = open(nf, 'rb')

    txt = f.readlines()
    txt = ' '.join(txt)

    f.close()
    return txt

def compteLignes(nf, fdl='\n', tbuf=16384):
    """Compte le nombre de lignes du fichier nf"""
    c = 0
    f = open(nf, 'rb')
    while True:
        buf = None
        buf = f.read(tbuf)
        if len(buf)==0:
            break
        c += buf.count(fdl)
    f.seek(-1, 2)
    car = f.read(1)
    if car != fdl:
        c += 1
    f.close()
    return c


# ## sac de mots

# In[4]:

def preprocessSent(s):
    punc = u''.join([ch for ch in string.punctuation]) # je laisse les points pour pouvoir séparer les phrases
    punc += u'\n\r\t\\'
    table =string.maketrans(punc + string.digits, ' '*(len(punc)+len(string.digits)))
    s = string.translate(unicodedata.normalize("NFKD",s).encode("ascii","ignore"),table).lower() # elimination des accents + minuscules
    return s
def count_ngrams(s,n=2,dic=None):
    if dic is None:
        dic=Counter()
    for i in range(len(s)):
        for j in range(i+1,min(i+n+1,len(s)+1)):
            dic[u''.join(s[i:j])]+=1
    return dic


# In[5]:

docProPre = [[] for i in range(len(docs))]
for i in range(len(docs)):
    for j in range(len(docs[i])):
        docProPre[i].append(preprocessSent(docs[i][j]))
    docProPre[i]=[s[:-1] for s in docProPre[i] if len(s)>2] # elimination des phrases/docs vides


# In[6]:

docProPre[0][0]


# In[7]:

indToDic = {}
dicToInd = {}
v = np.unique(np.array([a for doc in docProPre for w in doc for a in w.split()]))
for i,w in enumerate(v):
    
    dicToInd[i]=w
    indToDic[w]=i


# In[8]:

D = len(dicToInd)
docP = [docProPre[i][j].split() for i in range(len(docProPre)) for j in range(len(docProPre[i]))  ]
N=len(docP)



# In[9]:

bow=scipy.sparse.lil_matrix((N,D))
cpt=0
def add1(i,j):
    bow[i,j] += 1


# In[10]:


[[add1(i,indToDic[m]) for m in s if m in indToDic] for i,s in enumerate(docP)]

print "bow computed"


# In[11]:

bowN = bow / bow.sum(1)


# In[116]:

import numpy.random as rand
prototypes = bowN[rand.permutation(N)[:20]]


# In[57]:

sim = bowN.dot(prototypes.T) # métrique type cos
y= sim.argmax(1) # comprendre les dimensions des matrices pour comprendre ce calcul


# In[117]:

for ite in range(20):
    sim = bowN.dot(prototypes.T) # métrique type cos
    y= sim.argmax(1) # comprendre les dimensions des matrices pour comprendre ce calcul
    for i in np.unique(np.array(y)):
        prototypes[i] = bowN[np.where(y==i)[0]].sum(axis=1)/np.array(np.where(y==i)).shape[2]
    print 'itérations '+str(ite) +" check"


# In[138]:

ySur = [i for i,doc in enumerate(docs) for di in doc]


# In[161]:

purete = np.zeros((20))
for i in range(20):
    purete[i] = np.where(y==i)==np.where(ySur==i)
    


# In[123]:

np.unique(np.array(y))


# In[122]:

save = y


# In[128]:

purete = np.zeros(y.shape[0])


# In[146]:

print ySur[np.where(ySur==1)[0]]


# In[170]:

purete = np.zeros((20,20))
purete


# In[221]:

nb_classes=20
purete = np.zeros((20,2))

for c in range(nb_classes):
    index=np.where(y==c)[0]
    #print ySur[index]
    salut=np.bincount(ySur[index][0])
    m=np.max(salut)*100./np.array(np.where(y==c)).shape[2]
    
    
    
    
    purete[c] = np.argmax(salut),m
    
    #     print np.argmax([[1. for cl in range(20)] for i in range(len(ySur)) if(ySur==cl)])
    #for j in range(20):
        
    
print purete


# In[216]:

index = np.where(y==0)[0]
print np.bincount(ySur[index][0])*1./np.array(np.where(y==0)).shape[2]


# In[217]:

np.array(np.where(y==0)).shape[2]


# In[ ]:




# In[189]:

y


# In[ ]:




# In[ ]:



