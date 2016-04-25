
# coding: utf-8

#  # Template
#  
#  
#  ### Essai 1, Trop complexe
#  

# In[182]:

import numpy as np
import copy as cp

def createReport(t,c,l,s,dt):
    report = ""
#     phrase d'intro
    if(dt=="pourcentage"):
        report += "En "+str(c[-1])+" en "+str(l[0])+", le taux "+ findDeterminant(s) + " est de "+str(t[0][-1])+"%."
    
    return report


def findDeterminant(s):
    det = s.split(' ')[0]
    if(det == "le"):
        return "du "+str(s.split(' ')[1])
    if(det=="la"):
        return 'de la '+str(det[1])
    

sujet = "le chomage"
typeDonnee = "pourcentage" # differente options ici
tab = np.array([[10,10.2],[9.3,9.5],[11.4,11],[10.7,10.4],[8.9,8.7],[7.4,7.3],[8,7.7]])
colonnes = np.array([2014,2015])
lignes = np.array(["France","Normandie","Seine Maritime","Eure","Ile de France","Yvelines","Hauts de Seine"])

createReport(tab,colonnes,lignes,sujet,typeDonnee)


# ## Essai 2, 2 colonnes Années Label
# 

# In[219]:

import numpy as np
import copy as cp

def taux(a,b):
    return np.floor(100*np.abs(b-a)/float(b))

def tauxCroissance(a,b):
    return np.abs(np.floor((np.exp(np.log((float(a[1])/float(b[1]))) / (a[0]-b[0]))-1)*10000)/100)

# def variation_globale_pourcentage(tableau): 
    
#     len=tableau.shape[0]
        
#     r = taux(tableau[0][1],tableau[len-1][1]) if(tableau[0][1]>0) else tableau[len-1][1]
#     return r

def rupture_croissance(tableau): 

    copy = cp.copy(tableau)
    decalage = 1
    for i,e in enumerate(tableau[decalage:]):
        copy[i+decalage][1] = np.sign(e[1] - tableau[i][1])
        
    return np.where(copy == -1)[0]




def main(t,label,typeLabel,genreLabel,unite=None,genreUnite=None):
    print (phraseIntro(t,label,typeLabel,genreLabel,unite,genreUnite))
    print comparaisonDerniereAnnee(t,label,typeLabel,genreLabel,unite,genreUnite)
    print phraseVariationGlobale(t,label,typeLabel,genreLabel,unite=None,genreUnite=None)

def phraseIntro(t,label,typeLabel,genreLabel,unite=None,genreUnite=None):
    p = ""
    p+="En "+str(t[-1][0])+ " en France "
#     pourcentage
    if(typeLabel == "p"):
        p+="le taux "
        if(genreLabel == "f"):
            p+="de la "+label+" "
        if(genreLabel == "m"):
            p+="du "+label+" "
        p+="est de "+str(t[-1][1])+"%"
#     quantité
    if(typeLabel=="q"):
        if(t[-1][1]>1):
            p+="on compte "+str(t[-1][1])+" "+label+"s"
        if(t[-1][1]==1 and genreLabel=="f"):
            p+="on compte une "+label+""
        if(t[-1][1]==1 and genreLabel=="m"):
            p+="on compte un "+label+""
        if(t[-1][1]==0):
            p+="on compte zéro "+label+""
#     volume
    if(typeLabel=="v"):
        if(t[-1][1]>1):
            p+="le volumes des "+str(label)+"s s'élève à "+str(t[-1][1])+ " "+ str(unite)+"s"
        if(t[-1][1]==1 and genreUnite=="f"):
            p+="le volumes des "+str(label)+"s s'élève à une "+ str(unite)+""    
        if(t[-1][1]==1 and genreUnite=="m"):
            p+="le volumes des "+str(label)+"s s'élève à un "+ str(unite)+""    
        if(t[-1][1]==0):
            p+="le volumes des "+str(label)+"s s'élève à zéro "+ str(unite)+""    
    p+="."
    return p


def comparaisonDerniereAnnee(t,label,typeLabel,genreLabel,unite=None,genreUnite=None):
    p = ""
    if(t.shape[0]<2):
        return ""
    if(t[-2][1]==t[-1][1]):
        p+="Ce sont les même chiffres qu'en "+str(t[-2][0])+"."
        return p
    p+="C'est une "
    variation = taux(t[-2][1],t[-1][1])
    if(t[-2][1]<t[-1][1]):
        p+="augmentation de "
    if(t[-2][1]>t[-1][1]):
        p+="baisse de "
    p+=str(variation)+"% par rapport à "+str(t[-2][0])+" ("+str(t[-2][1])+" "+str(unite)
    if(t[-2][1]>1):
        p+="s)"
    else:
        p+=")"
    p+='.'
    return p


def phraseVariationGlobale(t,label,typeLabel,genreLabel,unite=None,genreUnite=None):
    if(t.shape[0]<3):
        return ""
    p = ""
    p += "Sur les "+str(t[-1][0]-t[0][0])+" dernières années ("+str(t[0][0])+"-"+str(t[-1][0])+") on remarque une "
    if(t[0][1]==t[-1][1]):
        p+="stagnation"
    variation = taux(t[0][1],t[-1][1])
    moy = tauxCroissance(t[-1],t[0])
    if(t[0][1]<t[-1][1]):
        p+="augmentation de "+str(variation)+"%, soit une augmentation moyenne de "+str(moy)+"% par an."
    if(t[0][1]>t[-1][1]):
        p+="baisse de "+str(variation)+"%, soit une baisse moyenne de "+str(moy)+"% par an."
    
    
    return p



# In[220]:

#pourcentage masculin
t = np.array([[2000,40],[2000,30],[2015,21]])
label = "chomage"
typeLabel = "p"
genreLabel = "m"
unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)


# In[221]:

#pourcentage féminin
t = np.array([[2014,18],[2015,20]])
label = "croissance"
typeLabel = "p"
genreLabel = "f"
unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)


# In[222]:

#quantité féminin
t = np.array([[2003,2],[2010,10],[2015,3]])
label = "femme"
typeLabel = "q"
genreLabel = "f"
unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)

#quantité féminin
t = np.array([[2003,2],[2010,10],[2015,1]])
label = "femme"
typeLabel = "q"
genreLabel = "f"

unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)


# In[223]:

#quantité masculins
t = np.array([[2014,23],[2015,23]])
label = "bidonville"
typeLabel = "q"
genreLabel = "m"
unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)

#quantité masculins
t = np.array([[2002,3],[2012,2],[2015,1]])
label = "bidonville"
typeLabel = "q"
genreLabel = "m"
unite=None
genreUnite=None
main(t,label,typeLabel,genreLabel,unite,genreUnite)


# In[224]:

#volume féminins
t = np.array([[2014,1],[2015,23]])
label = "exportation"
typeLabel = "v"
genreLabel = "f"
unite ="tonne"
genreUnite = "f"

main(t,label,typeLabel,genreLabel,unite,genreUnite)

#volume féminins
t = np.array([[2010,88],[2013,132],[2015,143]])
label = "exportation"
typeLabel = "v"
genreLabel = "f"
unite ="tonne"
genreUnite = "f"

main(t,label,typeLabel,genreLabel,unite,genreUnite)


# In[ ]:




# In[ ]:




# In[ ]:




# In[232]:

print rupture_croissance(np.array([[2010,88],[2013,60],[2014,45],[2015,13],[2016,143]]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



