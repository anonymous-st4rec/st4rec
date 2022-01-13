import torch
import torch.nn as nn

class ST4Rec_Adapative:
    def __init__(self, model,args):
        self.args = args
        self.model = model
        self.t = args.temperature
        self.le_lambda = args.le_lambda
        self.onehot =args.onehot

        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.batch_size = args.train_batch_size
        self.device = args.device

    def compute(self, batch):

        seqs, labels = batch
        pred, soft_target = self.model(seqs)
        pred = pred[:, -1, :]
        soft_target = soft_target[:, -1, :]
        labels = labels[labels>0]

        if self.onehot:
            onehot = torch.nn.functional.one_hot(labels, num_classes=soft_target.size(-1))
            tmp = self.kl((pred.softmax(dim=-1)/self.t).log(), ((soft_target.softmax(dim=-1)/self.t)+onehot).softmax(dim=-1)) 
        else:
            tmp = self.kl((pred.softmax(dim=-1)/self.t).log(), soft_target.softmax(dim=-1)/self.t)
        loss = (1-self.le_lambda)*self.ce(pred, labels) + pow(self.t,2)*self.le_lambda*tmp

        return loss

class ST4Rec_Mannual:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.t = args.temperature
        self.le_lambda = args.le_lambda
        self.onehot =args.onehot
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.batch_size = args.train_batch_size
        self.device = args.device
    def compute(self, batch):
        
        seqs, labels = batch
        pred,_ = self.model(seqs)  # B * L * N   
        pred = pred[:, -1, :]  
        labels = labels[labels>0]

        onehot = torch.nn.functional.one_hot(labels, num_classes=pred.size(-1)).to(torch.device(self.device)) 
        soft_target = onehot.float()
        soft_target *= 0.9
        soft_target[soft_target==0] = 0.1/(onehot.size(-1)-1)
      
        if self.onehot:
            soft_target = soft_target/self.t + onehot
            tmp = self.kl((pred.softmax(dim=-1)/self.t).log(), soft_target.softmax(dim=-1)) 
        else:
            tmp = self.kl((pred.softmax(dim=-1)/self.t).log(), soft_target/self.t) 

        loss = (1-self.le_lambda)*self.ce(pred, labels) + pow(self.t,2)*self.le_lambda*tmp
        return loss

class CE:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch
        pred,_ = self.model(seqs)  # B * L * N
        pred = pred[:, -1, :]  
        labels = labels[labels>0] 
        loss = self.ce(pred, labels)
        return loss

