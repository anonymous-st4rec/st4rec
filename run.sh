#!/bin/bash

#ce ST4Rec_Adapative ST4Rec_Mannual
python main.py \
--save_path /home/sasrecce \
--data_path /home/movielens30.csv \
--train_batch_size 256 \
--test_batch_size 256 \
--d_model 128 \
--max_len 30 \
--attn_heads 4 \
--sasrec_layers 16 \
--enable_res_parameter 1 \
--dropout 0.2 \
--eval_per_steps 3000 \
--num_epoch 30 \
--loss_type ce \
--device cuda:1 \
--le_lambda 0.9 \
--temperature 1 \
--onehot 1 \

