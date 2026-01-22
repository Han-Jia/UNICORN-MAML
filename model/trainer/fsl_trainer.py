import time
import os.path as osp
import numpy as np
from copy import deepcopy
import os
import pdb
import torch
import torch.nn.functional as F
from model.models.dyn_weight import read_data
from collections import OrderedDict
import matplotlib.pyplot as plt
import statistics
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer, get_cross_shot_dataloader, get_class_dataloader
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..','..'))


val_grad= torch.zeros(1,1)




class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.cov_layers= OrderedDict()
        self.cov= OrderedDict()
        self.high_acc = {}
        self.high_loss = {}
        self.grad_no =60
        self.cal=False
        self.evalBool = False
        self.best_reg = 0
        self.best_err = 1e6
        self.loss2 = 0
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.keys_grads= ['.'.join(k.split('.')[1:]) for k,v in self.model.named_parameters() if 'encoder' in k]# get the 50 keys for dictionary
       
        # save running statistics
        running_dict = {}
        for e in self.model.encoder.state_dict():
            if 'running' in e:
                key_name = '.'.join(e.split('.')[:-1])
                if key_name in running_dict:
                    continue
                else:
                    running_dict[key_name] = {}
                # find the position of BN modules
                component = self.model.encoder
                for att in key_name.split('.'):
                    if att.isdigit():
                        component = component[int(att)]
                    else:
                        component = getattr(component, att)
                
                running_dict[key_name]['mean'] = component.running_mean
                running_dict[key_name]['var'] = component.running_var
        self.running_dict = running_dict          
                
        # compute PCA given PCA-Noise
        if args.model_class == 'MAMLNoise' and args.noise_type == 'PCA':
            # get pre-trained features
            if osp.exists(osp.join(*args.para_init.split('/')[:-1], 'PCAStats-{}.dat'.format(args.backbone_class))):
                PCAStats = torch.load(osp.join(*args.para_init.split('/')[:-1], 'PCAStats-{}.dat'.format(args.backbone_class)))
                self.model.PCAStats = PCAStats
            else:
                self.class_loader = get_class_dataloader(args)
                self.model.eval()
                PCAStats = {}
                embedding_list = []
                for batch in tqdm(self.class_loader, desc='Get Embeddings', ncols=50):
                    if torch.cuda.is_available():
                        c_data, c_label = batch[0].cuda(), batch[-1].cuda()
                    else:
                        c_data, c_label = batch[0], batch[-1]
                    unique_c_label = torch.unique(c_label)
                    assert(unique_c_label.shape[0] == 1)
                    c_label = unique_c_label.item()
                    # split the data in to shots and add them to the corresponding queue
                    with torch.no_grad():
                        inst_emb = []
                        for j in range(int(np.ceil(c_data.shape[0] / 128))):
                            inst_emb.append(self.model.encoder(c_data[j*128:min((j+1)*128, c_data.shape[0]), :], embedding = True))
                        inst_emb = torch.cat(inst_emb)
                    embedding_list.append(inst_emb.cpu())    
                    
                # compute PCA
                whole_embedding = np.concatenate(embedding_list)
                from sklearn.decomposition import PCA
                # np.where(np.cumsum(pca.explained_variance_ratio_) > 0.9)
                pca = PCA(n_components=60) 
                pca.fit(whole_embedding)                
                PCAStats['s_values'] = torch.Tensor(pca.singular_values_)
                PCAStats['s_vectors'] = torch.Tensor(pca.components_)
                PCAStats['mean'] = torch.Tensor(pca.mean_)
                if torch.cuda.is_available():
                    PCAStats['s_values'] = PCAStats['s_values'].cuda()
                    PCAStats['s_vectors'] = PCAStats['s_vectors'].cuda()
                    PCAStats['mean'] = PCAStats['mean'].cuda()                
                self.model.PCAStats = PCAStats
                torch.save(PCAStats, osp.join(*args.para_init.split('/')[:-1], 'PCAStats-{}.dat'.format(args.backbone_class)))        
                self.model.train()
    
    
    def update_cov(self, cov_layers):
            for key,value in cov_layers.items():
                if key in self.cov:
                    self.cov[key] = self.cov[key] + value #torch.mean(torch.stack(value,dim=0),0, True) 
                else:
                    self.cov[key] = value


                      
    def prepare_label(self):
        # prepare one-hot label
        args = self.args
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        return label
    
    def train(self):
        layers_gradients = {}
        alpha=0.5
        for key in self.keys_grads:
            #print(key) no encoder.
            layers_gradients[key] = []

        args = self.args

        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        #g_dic={}
        #for k in self.keys_grads:
        #    g_dic[k]=[]
        # start FSL training
        label = self.prepare_label()

        for epoch in range(1, args.max_epoch + 1):
            # initialize the repo with embeddings
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1, tl2, ta = Averager(), Averager(), Averager()
            start_tm = time.time()
            self.model.zero_grad()
            #penality = 0

            for i, batch in enumerate(self.train_loader): #for batch in self.train_loader:
                self.train_step += 1
                data, gt_label = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                
                # check nan in data
                #print(data.isnan().any(), "data")    
                
                gt_label = gt_label[:args.way] # get the ground-truth label of the current episode
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                support = data[:args.way * args.shot]
                query = data[args.way * args.shot:]
                logits, cons_loss = self.model(support, query)
                #print(logits.isnan().any(), "logits")
                
                        

                loss = F.cross_entropy(logits, label)
                if args.cons_loss == 1:
                    loss = (loss +  cons_loss) * 0.5
                #print("loss",loss.data) #one value
   

                tl2.add(loss.item())
                #if self.evalBool == True and self.loss2!=0:
                    #cacluate the variance for the val tasks and penality the the outer loop loss
                #param= OrderedDict(self.model.named_parameters())
                #name_list, tensor_list = zip(*param.items())
                #penality += grads[0].pow(2).mean()#compute_irm_penalty(loss,tensor_list)

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc, _ = count_acc(logits, label)
                
             
    

                tl1.add(loss.item())
                ta.add(acc)
                


                loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)
                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)                    
                self.model.zero_grad()
                    
                self.try_logging(tl1, tl2, ta)
                # refresh start_tm
                start_tm = time.time()
            
            self.lr_scheduler.step()
            print('LOG: Epoch-{}: Train Acc-{}'.format(epoch, acc))
            self.try_evaluate(epoch)
            self.evalBool = True
                
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )


  
      
        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        global val_grad
        args = self.args
        args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
        args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query
        # evaluation mode
        #support_array = np.load(self.memory_loc,allow_pickle=True)
        #print("support_array : ", len(support_array))
        # if SD in validation not test : read_data(args)
        self.model.eval()
        # record the runing mean and variance before validation
        for e in self.running_dict:
            self.running_dict[e]['mean_copy'] = deepcopy(self.running_dict[e]['mean'])
            self.running_dict[e]['var_copy'] = deepcopy(self.running_dict[e]['var'])
            
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        for i, batch in enumerate(data_loader, 1):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]
            #calcualte variance for the val tasks
            #var = torch.var(data)#np.var(data.detach().cpu().numpy(),axis=0)

            support = data[:args.eval_way * args.eval_shot]
            query = data[args.eval_way * args.eval_shot:]
            logits = self.model.forward_eval(support, query, 'val')
            for e in self.running_dict:
                self.running_dict[e]['mean'] = deepcopy(self.running_dict[e]['mean_copy'])
                self.running_dict[e]['var'] = deepcopy(self.running_dict[e]['var_copy'])
                
            loss = F.cross_entropy(logits, label)
            
            acc,_ = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
            del data, support, query, logits, loss
            torch.cuda.empty_cache()
            #loss.backward
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        #self.optimizer.step
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
            self.model.encoder_repo.eval()

        args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query
        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
        args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query        
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        # for dynamic weight read files
        #read_data(args)
        
        # record the runing mean and variance before validation
        for e in self.running_dict:
            self.running_dict[e]['mean_copy'] = deepcopy(self.running_dict[e]['mean'])
            self.running_dict[e]['var_copy'] = deepcopy(self.running_dict[e]['var'])        
        record = np.zeros((100, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        for i, batch in tqdm(enumerate(self.test_loader, 1)):
            if torch.cuda.is_available():
                data = batch[0].cuda()
            else:
                data = batch[0]

            #data_Accum=torch.cat((data_Accum,data),0)
            #dat_var=np.mean(np.var(data_Accum.detach().cpu().numpy(),axis=0))
            support = data[:args.eval_way * args.eval_shot]
            query = data[args.eval_way * args.eval_shot:]

            # calling the forward_eval
            logits = self.model.forward_eval(support, query, 'test')
            for e in self.running_dict:
                self.running_dict[e]['mean'] = deepcopy(self.running_dict[e]['mean_copy'])
                self.running_dict[e]['var'] = deepcopy(self.running_dict[e]['var_copy'])

            # layers_gradients dictionay key:[] where key is the layer name and the [] contains its values
            
                    
            loss = F.cross_entropy(logits, label)
            acc,_ = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc
            del data, support, query, logits, loss
            torch.cuda.empty_cache()

            #if i==1000:
            #     break
            
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
    
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl
    
        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))
        
        args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query
        return vl, va, vap

    def evaluate_test_cross_shot(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        # record the runing mean and variance before validation
        for e in self.running_dict:
            self.running_dict[e]['mean_copy'] = deepcopy(self.running_dict[e]['mean'])
            self.running_dict[e]['var_copy'] = deepcopy(self.running_dict[e]['var'])        
        # num_shots = [1, 5, 10, 20, 30, 50]
        num_shots = [1, 5, 8]
        record = np.zeros((100, len(num_shots))) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
        for s_index, shot in enumerate(num_shots):
            test_loader = get_cross_shot_dataloader(args, shot)
            args.eval_shot = shot
            args.old_way, args.old_shot, args.old_query = args.way, args.shot, args.query
            args.way, args.shot, args.query = args.eval_way, args.eval_shot, args.eval_query        
            for i, batch in tqdm(enumerate(test_loader, 1)):
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data = batch[0]
    
                support = data[:args.eval_way * shot]
                query = data[args.eval_way * shot:]    
                logits = self.model.forward_eval(support, query, 'test')
                for e in self.running_dict:
                    self.running_dict[e]['mean'] = deepcopy(self.running_dict[e]['mean_copy'])
                    self.running_dict[e]['var'] = deepcopy(self.running_dict[e]['var_copy'])
                
                loss = F.cross_entropy(logits, label)
                acc,_ = count_acc(logits, label)
                record[i-1, s_index] = acc
                del data, support, query, logits, loss
                torch.cuda.empty_cache()
                 
                #if i==1000:
                #    break
            
            assert(i == record.shape[0])
            
            va, vap = compute_confidence_interval(record[:,s_index])
            print('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))
            args.way, args.shot, args.query = args.old_way, args.old_shot, args.old_query

        with open(osp.join(self.args.save_path, '{}+{}-CrossShot'.format(va, vap)), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))                
            for s_index, shot in enumerate(num_shots):
                va, vap = compute_confidence_interval(record[:,s_index])
                f.write('Shot {} Test acc={:.4f} + {:.4f}\n'.format(shot, va, vap))
                
    
    def final_record(self):
        # save the best performance in a txt file

        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                    self.trlog['max_acc'],
                    self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['test_acc'],
                    self.trlog['test_acc_interval']))            
