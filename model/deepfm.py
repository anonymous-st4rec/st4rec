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

class MLPLayers(nn.Module):
    r""" MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout, args):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout


        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(args.act_function)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        


    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class DeepFM(nn.Module):
    def __init__(self, args):
        super(DeepFM, self).__init__()
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

        self.num_features = args.num_item  #item size
        self.num_factors = args.d_model #embeding size
        self.act_function = args.act_function 
        self.layers = args.dfm_layers  
        self.drop_prob = args.drop_prob 
        self.maxlen = args.max_len
        self.embeddings = nn.Embedding(self.num_features+1, self.num_factors)
        self.biases = nn.Embedding(self.num_features+1, 1) 
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        FM_modules.append(nn.BatchNorm1d(self.num_factors))		
        FM_modules.append(nn.Dropout(self.drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)
        #FM的Linear-part 
        self.linear_part = nn.Linear(self.maxlen, 1)
        in_dim = self.num_factors*args.max_len
        self.layers = [in_dim] + self.layers 
        self.mlp_layers = MLPLayers(self.layers, self.drop_prob[-1], args)
        
        self.sigmoid = nn.Sigmoid()

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
        features *= timeline_mask 

        
        # Bi-Interaction layer
        sum_square_embed = features.sum(dim=1).pow(2)
        square_sum_embed = (features.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed) #[batch_size, embed_size]
        FM = self.FM_layers(FM) #[batch_size, embed_size]

       

        linear_part = self.linear_part(features.transpose(2,1)).reshape(FM.size(0),-1) #[batch_size, embed_size]
        FM = linear_part + FM  #[batch_size, embed_size]
        
        deep = self.mlp_layers(features.reshape(FM.size(0),-1)) 

        output = self.sigmoid(FM + deep)
        
        
        return output


    def forward(self,x):

        x = self.log2feats(x)
        

        return self.output(x), F.linear(x, self.embeddings.weight)  # B * L * D --> (B * L)* N 
    
