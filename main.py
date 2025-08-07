import random as rd
import os,csv
from prework import Glove_embedding
from comparison import NN_plot

file_path=os.path.join('CoNLL-2003','train.txt')
with open(file_path,'r') as f:
    line=f.readlines()
    n=len(line)
    train=[]
    for i in range(2,n):
        lines=line[i].split()
        train.append(lines)

file_path=os.path.join('CoNLL-2003','test.txt')
with open(file_path,'r') as f:
    line=f.readlines()
    n=len(line)
    test=[]
    for i in range(2,n):
        lines=line[i].split()
        test.append(lines)

file_path=os.path.join('..','task2','glove.6B','glove.6B.50d.txt')
with open(file_path,'rb') as f:
    lines=f.readlines()

n=len(lines)
trained_dict=dict()

for i in range(n):
    line=lines[i].split()
    trained_dict[line[0].upper()]=[float(line[j]) for j in range(1,51)]

def get(data):
    s,t=[],[]
    s1,t1=[],[]
    for x in data:
        if len(x)==0:
            if s1:
                s.append(s1)
                t.append(t1)
                s1,t1=[],[]
        else:
            s1.append(x[0])
            t1.append(x[1])
    if s1:
        s.append(s1)
        t.append(t1)

    c=[list(x) for x in zip(s,t)]
    return c

train=get(train)
test=get(test)

# for i in range(10):
#     print(test[i][0])

random_embedding=Glove_embedding(train, test)
random_embedding.get_id()

glove_embedding=Glove_embedding(train,test,trained_dict)
glove_embedding.get_id()

epochs=60
lr=0.001
batch_size=100

NN_plot(random_embedding,glove_embedding,50,50,lr,batch_size,epochs)







