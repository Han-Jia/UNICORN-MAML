import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.num_layers = 4
        # input layer
        self.add_module('{0}_{1}'.format(0,0), nn.Conv2d(x_dim, hid_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(0,1), nn.BatchNorm2d(hid_dim))
        # hidden layer
        for i in [1, 2]:
            self.add_module('{0}_{1}'.format(i,0), nn.Conv2d(hid_dim, hid_dim, 3, padding=1))   
            self.add_module('{0}_{1}'.format(i,1), nn.BatchNorm2d(hid_dim))     
        # last layer
        self.add_module('{0}_{1}'.format(3,0), nn.Conv2d(hid_dim, z_dim, 3, padding=1))   
        self.add_module('{0}_{1}'.format(3,1), nn.BatchNorm2d(z_dim))             
    
    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        output = x
        for i in range(self.num_layers):
            output = F.conv2d(output, params['{0}_{1}.weight'.format(i,0)], bias=params['{0}_{1}.bias'.format(i,0)], padding=1)
            output = F.batch_norm(output, weight=params['{0}_{1}.weight'.format(i,1)], bias=params['{0}_{1}.bias'.format(i,1)],
                                  running_mean=self._modules['{0}_{1}'.format(i,1)].running_mean,
                                  running_var=self._modules['{0}_{1}'.format(i,1)].running_var, training = self.training)
            output = F.relu(output)
            output = F.max_pool2d(output, 2)

        output = F.avg_pool2d(output, 5)     # AveragePool Here
        output = output.view(x.size(0), -1)
        
        if embedding:
            return output
        else:
            # Apply Linear Layer
            logits = F.linear(output, weight=params['fc.weight'], bias=params['fc.bias'])
            return logits
