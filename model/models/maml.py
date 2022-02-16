import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. """
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)         
    
    for _ in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    return updated_params

class MAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size) 
        else:
            raise ValueError('')

        self.args = args
        self.hdim = hdim
        self.encoder.fc = nn.Linear(hdim, args.way)

    def forward(self, data_shot, data_query):
        # update with gradient descent
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis
    
    def forward_eval(self, data_shot, data_query):
        # update with gradient descent
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query
    
    def forward_eval_perm(self, data_shot, data_query):
        # update with gradient descent
        # for permutation evaluation, and will output some statistics
        original_params = OrderedDict(self.named_parameters())
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_shot = self.encoder(data_shot, updated_params)
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
            
        # compute the norm of the params
        norm_list =  [torch.norm(updated_params[e] - original_params['encoder.' + e]).item() for e in updated_params.keys() ]
        return logitis_shot, logitis_query, np.array(norm_list) 
    
    
    def forward_eval_ensemble(self, data_shot, data_query_list):
        # update with gradient descent for Ensemble evaluation
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        # get shot accuracy and loss
        self.eval()
        logitis_query_list = []
        with torch.no_grad():
            for data_query in data_query_list:
                # logitis_shot = self.encoder(data_shot, updated_params)
                logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
                logitis_query_list.append(logitis_query)
        return logitis_query_list # logitis_shot,     