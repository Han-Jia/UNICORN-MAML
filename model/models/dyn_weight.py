# Example of getting neighbors for an instance
from decimal import InvalidContext
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.spatial as sp
import pdb
# example of calculating the softmax for a list of numbers
from scipy.special import softmax#softmax
import statistics
import torch
import os.path as osp
import time
import torch.nn.functional as F
import statistics
from memory_profiler import profile
from model.utils import (
    pprint, set_gpu,
    get_command_line_parser,
    postprocess_args, Timer
)

x_set=[]    
grads_set=[]




def read_data(args):
 
    THIS_PATH = osp.dirname(__file__)
    ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..','..'))
   
    

    #support_file=osp.join(ROOT_PATH,'unicorn',args.memory_dir,'Sup_memory.npy') unicorn for other dataset slake and miniimagenet
    grad_file=osp.join(ROOT_PATH,"unicorn",args.memory_dir,'Grad_memory.npy')

    with  open(grad_file, 'rb') as g_f:
        Read_sample=True
        try:
            while Read_sample:
            
                grad_sample = np.load(g_f,allow_pickle=True)
              
                grads_set.append(grad_sample)
        except:
            Read_sample= False#print("EoF")
            
        #
        # support_array = np.load(support_file,allow_pickle=True) #+str(i)
    

   

def calculate_Function(g1,g2,function_name): # g1 previous gradients, g2 current
    layer_scores={}
    
    for j, (p_g, c_g) in enumerate(zip(g1, g2)):
            
        c = np.array(c_g).flatten() # list of tensor
        p = np.array(p_g).flatten()
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

# Locate the most similar neighbors cosine
def get_neighbors(curr_grd, num_neighbors, stored_grads, dataset):
    distances = list()
    sim_list=[] # a list contains only the similarity values


    for i,grad in enumerate(stored_grads):
        #train_grads=[]
        if dataset == "Flower" and grad.ndim > 1:
            p_grad=np.array(np.squeeze(grad,0)) #(1,50) = (50) #.reshape(1, -1))
        else:
            p_grad=np.array(grad)

        avg_cos_sim= calculate_Function(p_grad, curr_grd, "cosine_sim")#statistics.mean(layer_scores['cosine'])#/len(curr_grd)##for all layers
        sim_list.append(avg_cos_sim)
        distances.append({'grads':grad,'sim': avg_cos_sim})
            
    sorted_sim = np.argsort(np.array(sim_list))[-num_neighbors:]
    neighbors = list()
    for i in sorted_sim:
        neighbors.append(distances[i])
    return neighbors

# Test distance function
#@profile
def calculate(curr_grads, args):
    k_num = args.similar_grads
    neighbors=[]
    
    neighbors = get_neighbors(curr_grads, k_num ,grads_set, args.dataset)
    

 
    sim_values = []
    grad_values = []

    
    for neighbor in neighbors:
        if neighbor['sim'] > 0:
            if args.dataset == "Flower" and neighbor['grads'].ndim > 1:
                Layers_differ_norm = calculate_Function(np.array(np.squeeze(neighbor['grads'],0)), curr_grads, "norm")
            else:
                Layers_differ_norm = calculate_Function(np.array(neighbor['grads']), curr_grads, "norm")
            diff_norm = np.mean(Layers_differ_norm)
            threshold = np.percentile(Layers_differ_norm, 75)
            if diff_norm <= threshold:
                sim_values.append(neighbor['sim'])
                grad_values.append(neighbor['grads']) 
    if len(sim_values) > 0:
        probabilities=softmax(sim_values)

        weighted_grads = []

        for weight, grad in zip(probabilities,grad_values):
            weighted_grads.append(np.multiply(weight,grad))


        dy_w = np.mean(weighted_grads,axis=0)
    else:
        dy_w = []

   
    
    return dy_w