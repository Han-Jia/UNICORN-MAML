import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from model.models.dyn_weight import calculate
from model.models.aug import augment
import itertools
import pdb
import statistics
import scipy.spatial as sp

def gradient_contrastive_loss(grad1, grad2, temperature=0.07, positive_pair=True):
    """
    Contrastive loss between two gradients.
    
    Args:
        grad1, grad2: Gradient tensors (can be from different samples/tasks) has all updated params
        temperature: Temperature scaling parameter
        positive_pair: Whether grad1 and grad2 should be similar (True) or dissimilar (False)
    """
    layer_losses=[]
    for p_g1, p_g2 in zip(grad1,grad2):
        # Flatten gradients if needed
        g1 = p_g1.flatten()
        g2 = p_g2.flatten()
    
        # Normalize gradients
        g1_norm = F.normalize(g1.unsqueeze(0), dim=1)
        g2_norm = F.normalize(g2.unsqueeze(0), dim=1)
    
        # Compute similarity
        similarity = torch.mm(g1_norm, g2_norm.T) / temperature
    
        if positive_pair:
        # Maximize similarity for positive pairs
            loss = -similarity
        else:
        # Minimize similarity for negative pairs
            loss = similarity

        layer_losses.append(loss.squeeze())
    return torch.mean(torch.Tensor(layer_losses))

def custom_activation(x):
    nx=[]
    for x_item in x:
        new_x= 1 / 4 * ((x_item + 1) ** 2)
        nx.append(new_x)
        #print("x_item: ",x_item)
        #print("new_x: ",new_x)
        #print("x_tem * new_x: ", x_item,"-", new_x,"-", x_item * new_x)
    return nx#torch.Tensor([1 / 4 * ((x_item + 1) ** 2) for x_item in x])

def calculate_Function(g1,g2,function_name): # g1 previous gradients, g2 current
    layer_scores={}
    
    for j, (p_g, c_g) in enumerate(zip(g1, g2)):
            
        c = np.array(c_g.detach().cpu()).flatten() # list of tensor
        p = np.array(p_g.detach().cpu()).flatten()
        if function_name == "cosine_sim":
            cos_sim = 1 - sp.distance.cosine(c, p)
            if j ==0:
                #layer_scores['layer']=[j]
                #layer_scores['shape']=[c_g.shape]
                layer_scores['cosine'] = [cos_sim]
            else:
                #layer_scores['layer'].append(j)
                #layer_scores['shape'].append(c_g.shape)
                layer_scores['cosine'].append(cos_sim)
        elif function_name == "norm":
            diff_norm = np.linalg.norm(c - p)
            base_norm = np.linalg.norm(p)
            current_norm = np.linalg.norm(c)
            if j ==0:
                layer_scores['norm'] = [diff_norm]
            else:
                layer_scores['norm'].append(diff_norm)

    if function_name == "cosine_sim": return statistics.mean(layer_scores['cosine'])
    else: return layer_scores['norm']

def _grad_norm(self, by=None, weight_adaptive=False):
        # shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
    if not by:
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p.data) if weight_adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        
    return norm

@torch.no_grad()
def get_cosine_matrix(mse_grads, ent_grads):
    # calculate inner product
    inner_prod = 0.0
    cosines=[]
    norms=[]
    for mse_g,ent_g in zip(mse_grads,ent_grads):
        #print("mse_g :" ,mse_g)
        #print("ent_g :" ,ent_g)
        if mse_g is not None and ent_g is not None:
            inner_prod = torch.sum(mse_g * ent_g)

            if ent_g.dim() <=2: 
                ent_norm =torch.linalg.norm(ent_g,ord=2)
                mse_norm = torch.linalg.norm(mse_g,ord=2)
                cosines.append(inner_prod/ (mse_norm * ent_norm + 1e-12))
            else:
                ent_norm = torch.linalg.matrix_norm(ent_g,ord=2)
                mse_norm = torch.linalg.matrix_norm(mse_g,ord=2)
                cosines.append(inner_prod/ (mse_norm * ent_norm + 1e-12))
        
        elif mse_g is None and ent_g is not None:
            mse_g=torch.ones_like(ent_g)
            inner_prod = torch.sum(mse_g * ent_g)
            if ent_g.dim() <=2: 
                ent_norm =torch.linalg.norm(ent_g,ord=2)
                mse_norm = torch.linalg.norm(mse_g,ord=2)
                cosines.append(inner_prod/ (mse_norm * ent_norm + 1e-12))
            else:
                ent_norm = torch.linalg.matrix_norm(ent_g,ord=2)
                mse_norm = torch.linalg.matrix_norm(mse_g,ord=2)
                cosines.append(inner_prod/ (mse_norm * ent_norm + 1e-12))
            
    return cosines

def get_cosine_value(mse_param, ent_param): # matrix for one param taken from the full grads
    # calculate inner product
    inner_prod = 0.0
    cos=0.0
    norms=[]
    
    ent_flat = torch.flatten(ent_param)
    if mse_param is not None and ent_param is not None:
        mse_flat = torch.flatten(mse_param)
        inner_prod += torch.sum(mse_param * mse_param)


    elif mse_param is None and ent_param is not None:
        mse_param=torch.ones_like(ent_param)
        mse_flat = torch.flatten(mse_param)
        inner_prod += torch.sum(mse_param * ent_param)
        
    ent_norm =torch.linalg.norm(mse_flat,ord=2)
    mse_norm = torch.linalg.norm(ent_flat,ord=2)

            
    cos = inner_prod/ (mse_norm * ent_norm + 1e-12)
            
    return cos

def return_grads(loss, params,  retain=False, allow=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, retain_graph=retain,  allow_unused=allow, create_graph=False)
    return  grads



def update_params(loss, params, grads_mse, mode = "train", step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order, allow_unused=True, retain_graph=True)
    
    updated_params = OrderedDict()
    

    cosines = []
    cus_lr = []
    if grads_mse != None:
        for mse_g, ent_g in zip(grads_mse,grads):
            cosines.append(get_cosine_value(mse_g,ent_g))
        cus_lr= custom_activation(cosines)
       
    if len(cus_lr) != 0:
        for name, param, grad, lr_scale in zip(name_list, tensor_list, grads, cus_lr):
            #if 'aug' not in name: 
            updated_params[name] = param - step_size *lr_scale *grad#*reshape(lr_scale,grad) * grad
    else:
        for name, param, grad in zip(name_list, tensor_list, grads):
            #if 'aug' not in name:
            #print(name, grad) 
            updated_params[name] = param - step_size  * grad

    return updated_params, grads



def query_update_params(loss, params, support_grads, args, mode = "train", step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order, allow_unused=True)
    
    cons_loss= gradient_contrastive_loss(grads,support_grads)
           
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad
        

    return updated_params,cons_loss



def inner_train_step(model, support_data, args):
    """ Inner training step procedure. """
 
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    

    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
        
    else:
        label = label.type(torch.LongTensor)   
         
    lbl_bool=[]

    #initial calculation for aux and ent before the inner loop
    #ent with original frist
    #augment
    aug_support = augment(support_data,0) 

    for i in range(args.inner_iters):
        feat = model(support_data, None, updated_params, embedding = True)  
             
        # _, ypred = model(aug_feat, None,  updated_params)
        # loss = F.cross_entropy(ypred, label)     
        # get features of augmented support data
        aug_feat, augpred = model(aug_support, None,  updated_params)#, embedding = True)    

        mse_loss= torch.nn.MSELoss()(aug_feat,feat)
        #mse_loss.backward(retain_graph=True)
        #grads_mse=[]
        #print("mse:")
        #for param in model.parameters():
            #print(param.grad)
        #    grads_mse.append(param.grad)
        grads_mse = return_grads(mse_loss, updated_params,retain=True, allow=True)
        
        loss = F.cross_entropy(augpred, label)  
        
        updated_params, ent_grads = update_params(loss, updated_params, grads_mse , mode = "train", step_size =  args.gd_lr, first_order=True)


    return updated_params, ent_grads, lbl_bool

def inner_test_step(model, support_data, args, mode_val_test):
    """ Inner training step procedure. """
    dy_w_dic = OrderedDict()
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)         
        dy_w_dic=None



    for _ in range(args.inner_iters):
        _, ypred = model(support_data, dy_w_dic, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params, _ = update_params(loss, updated_params, None, mode_val_test, step_size=args.gd_lr, first_order=True)
 
    
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
        #self.encoder.aug = nn.Linear(hdim, 1)
        self.dy_w_dic=None


    def forward(self, data_shot, data_query):
        # update with gradient descent
        updated_params, support_grads , bool_lbl = inner_train_step(self.encoder, data_shot, self.args)

        #flat_grad = torch.cat([g.contiguous().view(-1) for g in grads])
        _, logitis = self.encoder(data_query, self.dy_w_dic, updated_params) 
      
        # once evaluation on the query set to get the query gradeints
        label = torch.arange(self.args.eval_way).repeat(self.args.eval_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)  

        loss = F.cross_entropy(logitis, label)
        updated_params, con_loss = query_update_params(loss, updated_params,support_grads, self.args, "train", step_size=self.args.gd_lr, first_order=True)


        _, logitis = self.encoder(data_query, self.dy_w_dic, updated_params) #/ self.args.temperature

        return logitis,con_loss 
    
    def forward_eval(self, data_shot, data_query, mode_val_test):
        # update with gradient descent
        self.train()
        updated_params,support_grads , bool_lbl = inner_train_step(self.encoder, data_shot, self.args)
        

        # get shot accuracy and loss
        self.eval()

        _, logitis_query = self.encoder(data_query, self.dy_w_dic, updated_params) 

        return logitis_query
    
