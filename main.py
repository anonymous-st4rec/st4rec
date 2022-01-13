from dataset import TrainDataset, EvalDataset
from process import Trainer
from args import args
import pandas as pd
import torch.utils.data as Data
from model.SASRec import SASRec
from model.nfm import NFM
from model.deepfm import DeepFM

import sys
import os

sys.path.append(os.getcwd())



def main():
    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
   

    test_dataset = EvalDataset(args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')
    
    if args.model == 'sasrec' :
        model = SASRec(args)
    elif args.model == 'nfm':
        model = NFM(args)
    elif args.model == 'deepfm':
        model = DeepFM(args)
    else:
        raise NotImplementedError


    trainer = Trainer(args, model, train_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    main()
