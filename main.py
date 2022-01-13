from dataset import TrainDataset, EvalDataset
from train import Trainer
from args import args
import torch.utils.data as Data
from SASRec import SASRec


import sys
import os

sys.path.append(os.getcwd())



def main():
    train_dataset = TrainDataset(args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    val_dataset  = EvalDataset(args, mode='val')
    val_loader  = Data.DataLoader(val_dataset, batch_size=args.test_batch_size)

    test_dataset = EvalDataset(args, mode='test')
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print('dataset initial ends')
    
    model = SASRec(args)


    trainer = Trainer(args, model, train_loader, val_loader, test_loader)
    print('train process ready')

    trainer.train()


if __name__ == '__main__':
    main()
