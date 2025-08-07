import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Glove_embedding():
    def __init__(self,train,test,trained_dict=None):
        self.train=train
        self.test=test
        self.dict_words=dict()
        self.trained_dict=dict()
        if trained_dict:
            self.trained_dict=trained_dict
        self.train.sort(key=lambda x: len(x[0]))
        self.test.sort(key=lambda x: len(x[0]))
        self.train_x_matrix=[]
        self.train_y_matrix=[]
        self.test_x_matrix=[]
        self.test_y_matrix=[]
        self.embedding=[]
        self.tag_dict={'<pad>':0,'<begin>':1,'<end>':2}
        self.len_words=0
        self.longest=0
        self.len_tag=3

    def get_id(self):
        self.embedding.append([0]*50)
        for x in self.train:
            self.longest=max(self.longest,len(x[0]))
            for word in x[0]:
                if word not in self.dict_words:
                    self.dict_words[word]=len(self.dict_words)+1
                    if word not in self.trained_dict:
                        self.embedding.append([0]*50)
                    else:
                        self.embedding.append(self.trained_dict[word])

        for x in self.test:
            self.longest=max(self.longest,len(x[0]))
            for word in x[0]:
                if word not in self.dict_words:
                    self.dict_words[word]=len(self.dict_words)+1
                    if word not in self.trained_dict:
                        self.embedding.append([0]*50)
                    else:
                        self.embedding.append(self.trained_dict[word])
        for x in self.train:
            for word in x[1]:
                if word not in self.tag_dict:
                    self.tag_dict[word]=len(self.tag_dict)

        for x in self.test:
            for word in x[1]:
                if word not in self.tag_dict:
                    self.tag_dict[word]=len(self.tag_dict)

        self.len_words=len(self.dict_words)+1
        self.len_tag=len(self.tag_dict)

        for x in self.train:
            item=[self.dict_words[word] for word in x[0]]
            self.train_x_matrix.append(item)
            item=[self.tag_dict[word] for word in x[1]]
            self.train_y_matrix.append(item)

        for x in self.test:
            item=[self.dict_words[word] for word in x[0]]
            self.test_x_matrix.append(item)
            item=[self.tag_dict[word] for word in x[1]]
            self.test_y_matrix.append(item)

class ClsDataset(Dataset):
    def __init__(self,seq,tag):
        self.seq=seq
        self.tag=tag

    def __getitem__(self,idx):
        return self.seq[idx],self.tag[idx]

    def __len__(self):
        return len(self.seq)

def collate_fn(batch):
    seq,tag=zip(*batch)
    seqs=[torch.LongTensor(sent) for sent in seq]
    padded_seqs=pad_sequence(seqs, batch_first=True, padding_value=0)
    tags=[torch.LongTensor(t) for t in tag]
    padded_tags=pad_sequence(tags, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_seqs), torch.LongTensor(padded_tags)

def get_batch(x,y,batch_size):
    dataset=ClsDataset(x,y)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader






