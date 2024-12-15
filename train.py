"""
code of data parsing was adapted from https://github.com/EGiunchiglia/C-HMCNN

also ref to this repo https://github.com/minqukanq/hierarchical-multi-label-text-classification
"""

import os
os.environ["DATA_FOLDER"] = "./"

import argparse

import torch
import torch.utils.data
import torch.nn as nn

from utils.parser import *
from utils import datasets
import random
import sys
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc,precision_score, roc_curve, recall_score

from models.hmcn import HMCNF
from models.loss import Loss

# particular for CELLCYCLE_dataset,499
def split_globle_label_of_class(yg,num_classes_list=[99,178,142,77,4]):
    chunks = []
    start = 0
    for length in num_classes_list:
        end = start + length
        chunks.append(yg[:,start:end])
        start = end
    # yl[i], count number of "." is i
    return chunks

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train neural network')
    args = parser.parse_args(args=[])
    args.dataset = 'cellcycle_FUN'
    args.batch_size = 200
    args.lr = 1e-3
    args.hidden_dim = 400
    args.weight_decay = 1e-5
    args.num_epochs = 100
    args.drop_prob = 0.5
    args.seed = 421
    args.max_grad_norm = 0.1
    args.beta = 0.6
    args.threshold = 0.5

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Pick device
    args.device = 0
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]

   # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'church':27, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'hom':47034, 'seq':529, 'spo':86}
    output_dims_FUN = {'cellcycle':499, 'church':499, 'derisi':499, 'eisen':461, 'expr':499, 'gasch1':499, 'gasch2':499, 'hom':499, 'seq':499, 'spo':499}
    output_dims_GO = {'cellcycle':4122, 'church':4122, 'derisi':4116, 'eisen':3570, 'expr':4128, 'gasch1':4122, 'gasch2':4128, 'hom':4128, 'seq':4130, 'spo':4116}
    output_dims_others = {'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims = {'FUN':output_dims_FUN, 'GO':output_dims_GO, 'others':output_dims_others}

    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
        train.X, valX, train.Y, valY = train_test_split(train.X, train.Y, test_size=0.30, random_state=seed)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)
    # sum(different_from_0>0)/len depicts the sparsity
    # different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = np.uint8), dtype=torch.uint8)

    # Rescale dataset and impute missing data
    if ('others' in args.dataset):
        scaler = preprocessing.StandardScaler().fit((train.X))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X))
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)
        valX, valY = torch.tensor(scaler.transform(imp_mean.transform(valX))).to(device), torch.tensor(valY).to(device)
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X))) # concat.shape=[1628+848,77], scaler.mean_.shape=[77]
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X))) # scaler.mean_ - imp_mean.statistics_ =0;shape=77
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
        train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        

    # Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(valX, valY)]
    else:
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=False)

    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # model particlar for funcat_cellcycle
    model = HMCNF(num_classes_list=[99,178,142,77,4],total_classes=output_dims[ontology][data]+1,input_dim=input_dims[data],hidden_dim=args.hidden_dim,drop_prob=args.drop_prob,beta=args.beta)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
    criterion = Loss()
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.8)

    # Create folder for the dataset (if it does not exist)
    if not os.path.exists(os.path.join(os.environ["DATA_FOLDER"],'./logs/'+str(dataset_name)+'/')):
         fp = os.path.join(os.environ["DATA_FOLDER"],'./logs/'+str(dataset_name)+'/')
         os.makedirs(fp)
    # log file
    for epoch in range(args.num_epochs):
        model.train()
        for i, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            _,Yhat,Ylhat = model(x.float())
            local_labels = split_globle_label_of_class(labels)
            loss = criterion(labels,local_labels,Yhat,Ylhat)
            # Getting gradients w.r.t. parameters
            loss.backward()

            # gradient will not exceed 0.1=args.max_grad_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # Updating parameters
            optimizer.step()
            lr_schedule.step()
        print('Epoch: {} - Loss: {:.4f}'.format(epoch,loss))
        
        model.eval()
        predict_score_list = []
        val_loss_list = []
        true_label_list = []
        for i, (x,y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                score,Yhat,Ylhat = model(x.float())
                local_y= split_globle_label_of_class(y)
                valloss = criterion(y,local_y,Yhat,Ylhat)
                
                predict_score_list.append(score.cpu().numpy())
                val_loss_list.append(valloss.item())
                true_label_list.append(y.cpu().numpy())
        predict_score = np.concatenate(predict_score_list,axis=0)
        true_label = np.concatenate(true_label_list,axis=0)
        val_loss = np.average(val_loss_list)
        predict_onehot = (predict_score>= args.threshold).astype(int)
        eval_auc = roc_auc_score(y_true=true_label,y_score=predict_onehot,average='micro')
        eval_prc = average_precision_score(y_true=true_label,y_score=predict_onehot,average='micro')
        # eval_auc = roc_auc_score(y_true=true_label,y_score=predict_score,average='micro')
        # eval_prc = average_precision_score(y_true=true_label,y_score=predict_score,average='micro')
        # eval_precision = precision_score(y_true=true_label,y_pred=predict_onehot,average='micro')
        # eval_recall = recall_score(y_true=true_label,y_pred=predict_onehot,average='micro')
        # print('Epoch: {} - Loss: {:.4f}, precision: {:.5f} recall: {:.5f}\n'.format(epoch,val_loss,eval_precision,eval_recall))
        
        fp = os.path.join(os.environ["DATA_FOLDER"],'logs/'+str(dataset_name)+'/measures_batch_size_'+str(args.batch_size)+'_lr_'+str(args.lr)+'_weight_decay_'+str(args.weight_decay)+'_seed_'+str(args.seed)+'_num_layers_'+'._hidden_dim_'+str(args.hidden_dim))
        floss= open(fp,'a')
        floss.write('Epoch: {} - Loss: {:.4f}, AUC: {:.5f} PRC: {:.5f}\n'.format(epoch,val_loss,eval_auc,eval_prc))
        floss.close()
        print('Epoch: {} - Loss: {:.4f}, AUC: {:.5f} PRC: {:.5f}'.format(epoch,val_loss,eval_auc,eval_prc))
    # torch.save(model,'./myHMCN/m.pth')

if __name__ == "__main__":
    main()