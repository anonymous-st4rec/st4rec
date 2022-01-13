import torch
import pandas as pd
import torch.utils.data as Data

class TrainDataset(Data.Dataset):
    def __init__(self, args):
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        if args.eval_per_steps == 0:
            args.eval_per_steps = self.num_user//args.train_batch_size
        self.mask_token = self.num_item + 1
        self.max_len = args.max_len
        self.device = args.device
    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, -self.max_len - 3:-3].tolist()
        labels = [self.data[index,-3].tolist()]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(labels)


class EvalDataset(Data.Dataset):
    def __init__(self,  args, mode):
        self.data = pd.read_csv(args.data_path, header=None).replace(-1,0).values
        self.num_item = self.data.max()
        self.num_user = self.data.shape[0]
        self.max_len = args.max_len
        self.device = args.device
        self.mode = mode

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.data[index, :-2] if self.mode == 'val' else self.data[index, :-1]
        pos = self.data[index, -2] if self.mode == 'val' else self.data[index, -1]
        seq = list(seq)
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        answers = [pos]
        
        return torch.LongTensor(seq), torch.LongTensor(answers)
        
