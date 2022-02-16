import argparse
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler,RandomSampler
from model.utils import pprint, set_gpu, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
from tqdm import tqdm
from model.utils import one_hot
from copy import deepcopy
from itertools import permutations
import pickle

np.random.seed(0)
torch.manual_seed(0)

''' Evaluate MAML-Type Methods for ALL 120 permutations, and some statistics are saved'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='MAML', 
                            choices=['U2S', 'MAML', 'ProtoMAML'])     
    parser.add_argument('--backbone_class', type=str, default='Res12', 
                        choices=['Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    parser.add_argument('--model_path', type=str, default='./MAML-1-shot.pth')    
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--shot_list', type=str, default='1,5')   
    parser.add_argument('--gd_lr', default=0.001, type=float, 
                            help='The inner learning rate for MAML-Based model')        
    parser.add_argument('--inner_iters', default=5, type=int, 
                        help='The inner iterations for MAML-Based model')  
        
    # parser.add_argument('--fsl', action='store_true', default=False)                  # test FSL or not
    args = parser.parse_args()
    args.shot = 1
    args.orig_imsize = -1
    args.n_view = 1
    args.fix_BN = False
    args.way = 5
    args.temperature = 1
    pprint(vars(args))
    set_gpu(args.gpu)
    
    # Dataset and Data Loader
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5   
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5                       
    else:
        raise ValueError('Non-supported Dataset.')
        
    trainset = Dataset('train', args) 
    testset = Dataset('test', args) 
    args.num_class = trainset.num_class                  
    
    # shot = [1, 5, 10, 20, 30, 50]
    # FSL Test, 1-Shot, 5-Way
    num_shots = [int(e) for e in args.shot_list.split(',')]
    pemute_list = list(permutations(range(5), 5))
    test_acc_record_shot = np.zeros((2000, len(pemute_list), len(num_shots)))
    test_loss_record_shot = np.zeros((2000, len(pemute_list), len(num_shots)))
    test_acc_record_query = np.zeros((2000, len(pemute_list), len(num_shots)))
    test_loss_record_query = np.zeros((2000, len(pemute_list), len(num_shots)))    
    for shot_ind, shot in enumerate(num_shots):
        few_shot_sampler = CategoriesSampler(testset.label, 2000, 5, shot + 15)
        few_shot_loader = DataLoader(dataset=testset, batch_sampler=few_shot_sampler, num_workers=4, pin_memory=True)              
    
        shot_logit_list = np.zeros((2000, len(pemute_list), args.way  * shot, args.way))
        query_logit_list = np.zeros((2000, len(pemute_list), args.way  * 15, args.way))
        update_norm_list = np.zeros((2000, len(pemute_list), 50))
    
        label_shot, label_query = torch.arange(args.way).repeat(shot).long(), torch.arange(args.way).repeat(15).long()
        if torch.cuda.is_available():
            label_shot = label_shot.cuda()
            label_query = label_query.cuda()
            
        # Get Model
        if args.model_class == 'MAML':
            from model.models.maml import MAML 
            model = MAML(args)
        else:
            raise ValueError('No Such Model')
            
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path, map_location='cpu')['params']
        pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.train()
        model.args.shot = shot
        if torch.cuda.is_available():            
            torch.backends.cudnn.benchmark = True
            model = model.cuda()
            
        support_emb = []
        for i, batch in tqdm(enumerate(few_shot_loader), ncols=50, desc='1-Shot 5-Way Test'):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
            support, query = data[:5 * shot].view(shot, 5, 3, 84, 84), data[5 * shot:].view(15, 5, 3, 84, 84)       
            for p_index, p_value in enumerate(pemute_list):
                support_c, query_c = support[:, p_value, :, :, :].view(-1, 3, 84, 84), query[:, p_value, :, :, :].view(-1, 3, 84, 84)
                model.load_state_dict(model_dict)     
                if p_index == 0:
                    with torch.no_grad():
                        support_emb.append(model.encoder(support_c, embedding = True).detach().cpu())
                
                model.load_state_dict(model_dict)
                logits_shot, logits_query, updated_norm = model.forward_eval_perm(support_c, query_c)
                
                shot_acc = count_acc(logits_shot, label_shot)
                query_acc = count_acc(logits_query, label_query)
                
                test_acc_record_shot[i, p_index, shot_ind] = shot_acc
                test_loss_record_shot[i, p_index, shot_ind] = F.cross_entropy(logits_shot, label_shot).item()                
                test_acc_record_query[i, p_index, shot_ind] = query_acc
                test_loss_record_query[i, p_index, shot_ind] = F.cross_entropy(logits_query, label_query).item()
                
                shot_logit_list[i, p_index, :, :] = logits_shot.detach().cpu()
                query_logit_list[i, p_index, :, :] = logits_query.detach().cpu()
                update_norm_list[i, p_index, :] = updated_norm
                
                del support_c, query_c, logits_shot, logits_query
                torch.cuda.empty_cache()            
    
            del support, query
            torch.cuda.empty_cache()            
            
        current_acc_record_shot = test_acc_record_shot[:,:, shot_ind]
        current_loss_record_shot = test_loss_record_shot[:,:, shot_ind]        
        current_acc_record_query = test_acc_record_query[:,:, shot_ind]
        current_loss_record_query = test_loss_record_query[:,:, shot_ind]           
        # save current record
        savename = args.model_path.split('.')[-2].split('/')[-1].strip()
        with open('{}-Test-{}-Shot.pkl'.format(savename, shot), 'wb') as f:
            pickle.dump([current_acc_record_shot, current_loss_record_shot, current_acc_record_query, current_loss_record_query], f)
        
        with open('{}-Test-{}-Shot-EMB.pkl'.format(savename, shot), 'wb') as f:
            pickle.dump(support_emb, f)        
            
        with open('{}-Test-{}-logit.pkl'.format(savename, shot), 'wb') as f:
            pickle.dump([shot_logit_list, query_logit_list], f)
            
        with open('{}-Test-{}-updated_norm.pkl'.format(savename, shot), 'wb') as f:
            pickle.dump(update_norm_list, f)        
            
        m1, pm1 = compute_confidence_interval(current_acc_record_query.mean(-1))
        print('Shot-{}: Test Acc - {:.5f} + {:.5f}'.format(shot, m1, pm1))
        
        # compute variance
        max_value = current_acc_record_query.max(-1)
        min_value = current_acc_record_query.min(-1)
        m2, pm2 = compute_confidence_interval(max_value)
        m3, pm3 = compute_confidence_interval(min_value)        
        print('Shot-{}: Test Max Acc - {:.5f} + {:.5f}'.format(shot, m2, pm2))        
        print('Shot-{}: Test Min Acc - {:.5f} + {:.5f}'.format(shot, m3, pm3))           
        
    for shot_ind, shot in enumerate(num_shots):
        current_acc_record = test_acc_record_query[:,:, shot_ind]
        m1, pm1 = compute_confidence_interval(current_acc_record.mean(-1))
        print('Shot-{}: Test Acc - {:.5f} + {:.5f}'.format(shot, m1, pm1))
        
        # compute variance
        max_value = current_acc_record.max(-1)
        min_value = current_acc_record.min(-1)
        m2, pm2 = compute_confidence_interval(max_value)
        m3, pm3 = compute_confidence_interval(min_value)        
        print('Shot-{}: Test Max Acc - {:.5f} + {:.5f}'.format(shot, m2, pm2))        
        print('Shot-{}: Test Min Acc - {:.5f} + {:.5f}'.format(shot, m3, pm3))        