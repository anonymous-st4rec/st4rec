import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, uniform_, constant_
import numpy as np

def activation_layer(activation_name='relu'):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation_name.lower() == 'tanh':
            activation = nn.Tanh()
        elif activation_name.lower() == 'relu':
            activation = nn.ReLU()
        elif activation_name.lower() == 'leakyrelu':
            activation = nn.LeakyReLU()
        # elif activation_name.lower() == 'dice':
        #     activation = Dice(emb_dim)
        elif activation_name.lower() == 'none':
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError("activation function {} is not implemented".format(activation_name))

    return activation


class NFM(nn.Module):
    def __init__(self, args):
        super(NFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """
        self.model = args.model
        self.device = args.device

        self.num_features = args.num_item  
        self.num_factors = args.d_model 
        self.act_function = args.act_function 
        self.layers = args.nfm_layers  
        self.drop_prob = args.drop_prob 
        self.maxlen = args.max_len
        self.embeddings = nn.Embedding(self.num_features+1, self.num_factors)
        self.biases = nn.Embedding(self.num_features+1, 1) 
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        FM_modules.append(nn.BatchNorm1d(self.num_factors))		
        FM_modules.append(nn.Dropout(self.drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = self.num_factors 
        for dim in self.layers: 
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
            MLP_module.append(nn.BatchNorm1d(out_dim))
            MLP_module.append(activation_layer(args.act_function))
            MLP_module.append(nn.Dropout(self.drop_prob[-1]))
            
        self.deep_layers = nn.Sequential(*MLP_module)
        self.linear_part = nn.Linear(self.maxlen, 1)

        predict_size = self.layers[-1] if self.layers else self.num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        # Output
        self.output = nn.Linear(self.num_factors, self.num_features+1)
       

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / self.num_features)
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def log2feats(self, item_seq): #features [batch_size, seq_len, embed_size]

        features = self.embeddings(item_seq)  # [batch_size, seq_len, embed_size]
        timeline_mask = ~(item_seq == 0).unsqueeze(-1)
        features *= timeline_mask # broadcast in last dim  [batch_size, seq_len, embed_size]

        # Bi-Interaction layer
        sum_square_embed = features.sum(dim=1).pow(2)
        square_sum_embed = (features.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed) #[batch_size, embed_size]
        FM = self.FM_layers(FM) #[batch_size, embed_size]


        linear_part = self.linear_part(features.transpose(2,1)).reshape(FM.size(0),-1) #[batch_size, embed_size]
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        

        return linear_part + FM

    

    def forward(self,x):

        x = self.log2feats(x)

        return self.output(x), F.linear(x, self.embeddings.weight)  # B * L * D --> (B * L)* N 
    
