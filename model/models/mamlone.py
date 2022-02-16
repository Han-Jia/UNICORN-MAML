import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# UNICORN-MAML

def update_params(loss, params, acc_gradients, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad
        # accumulate gradients for final updates
        if name == 'fc.weight':
            acc_gradients[0] = acc_gradients[0] + grad
        if name == 'fc.bias':
            acc_gradients[1] = acc_gradients[1] + grad

    return updated_params, acc_gradients

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['fc.weight']), torch.zeros_like(updated_params['fc.bias'])]
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)        
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)
    return updated_params, acc_gradients

class MAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size) 
        else:
            raise ValueError('')
        
        self.args = args
        self.hdim = hdim
        self.encoder.fc = nn.Linear(self.hdim, args.way)        
        self.fcone = nn.Linear(self.hdim, 1)
        
    def forward(self, data_shot, data_query):
        # set the initial classifier
        self.encoder.fc.weight.data = self.fcone.weight.data.repeat(self.args.way, 1)
        self.encoder.fc.bias.data = self.fcone.bias.data.repeat(self.args.way)
        
        # update with gradient descent
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, self.args)
        
        # reupate with the initial classifier and the accumulated gradients
        updated_params['fc.weight'] = self.fcone.weight.repeat(self.args.way, 1) - self.args.gd_lr * acc_gradients[0]
        updated_params['fc.bias'] = self.fcone.bias.repeat(self.args.way) - self.args.gd_lr * acc_gradients[1]
        
        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis
    
    def forward_eval(self, data_shot, data_query):
        # set the initial classifier
        self.encoder.fc.weight.data = self.fcone.weight.data.repeat(self.args.way, 1)
        self.encoder.fc.bias.data = self.fcone.bias.data.repeat(self.args.way)
        
        # update with gradient descent
        self.train()
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, self.args)
        
        # reupate with the initial classifier and the accumulated gradients
        updated_params['fc.weight'] = self.fcone.weight.repeat(self.args.way, 1) - self.args.gd_lr * acc_gradients[0]
        updated_params['fc.bias'] = self.fcone.bias.repeat(self.args.way) - self.args.gd_lr * acc_gradients[1]
        
        self.eval()
        with torch.no_grad():        
            # logitis_shot = self.encoder(data_shot, updated_params)
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query # logitis_shot