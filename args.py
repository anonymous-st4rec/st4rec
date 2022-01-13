import argparse
import time
import os
import json
import pandas as pd
import torch
parser = argparse.ArgumentParser()

# train set args

parser.add_argument('--loss_type', type=str, default='ST4Rec_Adapative', choices=['ce', 'ST4Rec_Adapative','ST4Rec_Mannual']) 
parser.add_argument('--lr_decay_rate', type=float, default=1)
parser.add_argument('--lr_decay_steps', type=int, default=1250)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--metric_ks', type=list, default=[10,20]) 
parser.add_argument('--best_metric', type=str, default='NDCG@10')
parser.add_argument('--data_path', type=str, default="/home//movielens30.csv")
parser.add_argument('--save_path', type=str, default='None')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--max_len', type=int, default=30) 
parser.add_argument('--train_batch_size', type=int, default=256) 
parser.add_argument('--test_batch_size', type=int, default=256)

#2 model args 
parser.add_argument('--model', type=str, default='deepfm', choices=['sasrec','nfm','deepfm']) 
parser.add_argument('--d_model', type=int, default=128) 
parser.add_argument('--eval_per_steps', type=int, default=1000) 
parser.add_argument('--enable_res_parameter', type=int, default=1)  


#3 sasrec args
parser.add_argument('--attn_heads', type=int, default=4) 
parser.add_argument('--dropout', type=float, default=0.2)  
parser.add_argument('--d_ffn', type=int, default=512) 
parser.add_argument('--sasrec_layers', type=int, default=16) 


# FM超参
parser.add_argument('--nfm_layers', type=list, default=[128]) 
parser.add_argument('--dfm_layers', type=list, default=[128,128]) 
parser.add_argument('--drop_prob', type=list, default=[0.2,0.2]) 
parser.add_argument('--act_function', default='relu', type=str , help='activate function of mlp')



#4  label enhance
parser.add_argument('--temperature', type=float, default=1) 
parser.add_argument('--onehot', type=int, default=1) 
parser.add_argument('--le_lambda', type=float, default=0.9) 


args = parser.parse_args()


# other args

DATA = pd.read_csv(args.data_path, header=None)
num_item = DATA.max().max()
del DATA
args.num_item = int(num_item)

if args.save_path == 'None':
    loss_str = args.loss_type
    path_str = 'Model-' + args.model +'_le_lambda-' + str(args.le_lambda)+'-temperature-' + str(args.temperature)
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()