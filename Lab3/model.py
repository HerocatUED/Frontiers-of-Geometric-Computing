import torch
import torch.nn as nn
import numpy as np


class MLPnet(nn.Module):
    def __init__(self, in_dim = 3, hidden_dim = 512, hidden_num = 4, radius_init = 1, beta=100):
        super().__init__()
        # activate function
        self.activation = nn.Softplus(beta=beta)
        # build MLP
        layer_list = []
        # head
        head_lin = nn.Linear(in_dim, hidden_dim)
        torch.nn.init.constant_(head_lin.bias, 0.0)
        torch.nn.init.normal_(head_lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
        layer_list.extend([head_lin, self.activation])
        # hidden layers
        for i in range(hidden_num):
            lin = nn.Linear(hidden_dim, hidden_dim)
            # geometric initialization
            if i == hidden_dim - 2:
                torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(hidden_dim), std=0.00001)
                torch.nn.init.constant_(lin.bias, -radius_init)
            else:
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_dim))
                torch.nn.init.constant_(lin.bias, 0.0)   
            layer_list.extend([lin, self.activation])
        # tail
        tail_lin = nn.Linear(hidden_dim, 1)
        torch.nn.init.normal_(tail_lin.weight, 0.0, np.sqrt(2) / np.sqrt(int(in_dim/3)))
        torch.nn.init.constant_(tail_lin.bias, 0.0)
        layer_list.extend([tail_lin])
        # MLP network
        self.body = nn.Sequential(*layer_list)


    def forward(self, input):
        y = self.body(input)
        return y