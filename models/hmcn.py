import torch
import torch.nn as nn
import numpy as np

# funcat_cellcycle 5 hierarchical layers: [99,178,142,77,4]=>500
class HMCNF(nn.Module):
    def __init__(self,num_classes_list,total_classes,input_dim,hidden_dim,beta=0.5,drop_prob = 0.7):
        super(HMCNF,self).__init__()
        self.beta = beta
        self.g1fc = nn.Linear(input_dim,hidden_dim)
        self.g2fc = nn.Linear(hidden_dim,hidden_dim)
        self.g2fc_rs = nn.Linear(input_dim,hidden_dim)
        self.g3fc = nn.Linear(hidden_dim,hidden_dim)
        self.g3fc_rs = nn.Linear(input_dim,hidden_dim)
        self.g4fc = nn.Linear(hidden_dim,hidden_dim)
        self.g4fc_rs = nn.Linear(input_dim,hidden_dim)
        self.g5fc = nn.Linear(hidden_dim,hidden_dim)
        self.g5fc_rs = nn.Linear(input_dim,hidden_dim)
        self.gout = nn.Linear(hidden_dim,total_classes)
        
        self.l1fc = nn.Linear(hidden_dim,hidden_dim)
        self.l1out = nn.Linear(hidden_dim,num_classes_list[0])
        self.l2fc = nn.Linear(hidden_dim,hidden_dim)
        self.l2out = nn.Linear(hidden_dim,num_classes_list[1])
        self.l3fc = nn.Linear(hidden_dim,hidden_dim)
        self.l3out = nn.Linear(hidden_dim,num_classes_list[2])
        self.l4fc = nn.Linear(hidden_dim,hidden_dim)
        self.l4out = nn.Linear(hidden_dim,num_classes_list[3])
        self.l5fc = nn.Linear(hidden_dim,hidden_dim)
        self.l5out = nn.Linear(hidden_dim,num_classes_list[4])

        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
    #x.shape=[batch,arrt]
    def forward(self,x):
        gA1 = self.activ(self.g1fc(x))
        gA2 = self.activ(self.g2fc(gA1)+self.g2fc_rs(x))
        gA3 = self.activ(self.g2fc(gA2)+self.g3fc_rs(x))
        gA4 = self.activ(self.g2fc(gA3)+self.g4fc_rs(x))
        gA5 = self.activ(self.g2fc(gA4)+self.g5fc_rs(x))
        gA_drop = self.dropout(gA5)
        gP = nn.Sigmoid()(self.gout(gA_drop))

        lP1 = nn.Sigmoid()(self.l1out(self.activ(self.l1fc(gA1))))
        lP2 = nn.Sigmoid()(self.l2out(self.activ(self.l2fc(gA2))))
        lP3 = nn.Sigmoid()(self.l3out(self.activ(self.l3fc(gA3))))
        lP4 = nn.Sigmoid()(self.l4out(self.activ(self.l4fc(gA4))))
        lP5 = nn.Sigmoid()(self.l5out(self.activ(self.l5fc(gA5))))

        lP = torch.concat((lP1,lP2,lP3,lP4,lP5),dim=-1)

        output = self.beta*lP + (1-self.beta)*gP
        return output,gP,(lP1,lP2,lP3,lP4,lP5)





        