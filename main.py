import numpy as np
import pandas as pd
import pickle
from svm import develop
import os
from getEmbeddings import getCurrentEmbeddings
from gensim.models import Doc2Vec

if not os.path.isfile('./finalized_model.pkl'):
    develop()
print("Here")
#Loading the saved SVM model
filename = 'finalized_model.pkl'
clf = pickle.load(open(filename, 'rb'))
text_model=Doc2Vec.load("doc2vec.model") #Loading Doc2Vec Model
print("Loaded\n")
'''
xte = np.load('./xte.npy')
yte = np.load('./yte.npy')
y_pred = clf.predict(xte)
print(xte[2])
print(yte[2],type(y_pred[2]),y_pred[2])
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%
'''

t=int(input("Enter number of News Entries "))
for i in range(t):
    title=input("Enter Title ")
    author=input("Enter Author ")
    text=input("Enter News Content [In one single paragraph] ")
    if title=="" or author=="" or text=="":
        print("\nPlease Fill Complete details \n")
        continue
    d={'title':[title],'author':[author],'text':[text]}
    df=pd.DataFrame(data=d)

    emb_text=getCurrentEmbeddings(df)
    pk=np.zeros((1,300))
    pk[0]=text_model.infer_vector(emb_text)
    print(pk[0])
    y_pred = clf.predict(pk)
    print(y_pred)
    if y_pred[0]==1:
        print("\nFake News \n")
    elif y_pred[0]==0:
        print("\nReliable News \n")
    
    


