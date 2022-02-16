import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from model.networks.dropblock import DropBlock

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out
    
    def block_forward_para(self, x, params, base, mode, modules, downsample=False):
        '''the forard function of BasicBlock give parametes'''
        self.num_batches_tracked += 1
        
        residual = x
        
        out = F.conv2d(x, params[base + 'conv1.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn1.weight'], bias=params[base + 'bn1.bias'],
                           running_mean=modules['bn1'].running_mean,
                           running_var=modules['bn1'].running_var, training = mode)
        out = self.relu(out)
        
        out = F.conv2d(out, params[base + 'conv2.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn2.weight'], bias=params[base + 'bn2.bias'],
                               running_mean=modules['bn2'].running_mean,
                               running_var=modules['bn2'].running_var, training = mode)        
        out = self.relu(out)
        
        out = F.conv2d(out, params[base + 'conv3.weight'], stride=(1, 1), padding=(1, 1))
        out = F.batch_norm(out, weight=params[base + 'bn3.weight'], bias=params[base + 'bn3.bias'],
                               running_mean=modules['bn3'].running_mean,
                               running_var=modules['bn3'].running_var, training = mode)  
        
        if downsample is True:
            residual = F.conv2d(x, params[base + 'downsample.0.weight'], stride=(1, 1))
            residual = F.batch_norm(residual, weight=params[base + 'downsample.1.weight'], 
                                    bias=params[base + 'downsample.1.bias'],
                                    running_mean=modules['downsample']._modules['1'].running_mean,
                                    running_var=modules['downsample']._modules['1'].running_var, training = mode)
        out += residual
        out = F.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
                
        return out

#def forward_layer(x, params, base, mode, modules):
    ## forward of a layer given parameters
    #x = block_forward_para(x, params, base + '.0.', mode, modules['0']._modules, True)
    #return x


class ResNetMAMLUS(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=10):
        self.inplanes = 3
        super(ResNetMAML, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        x = self.layer1[0].block_forward_para(x, params, 'layer1' + '.0.', self.training, self._modules['layer1']._modules['0']._modules, True)
        x = self.layer2[0].block_forward_para(x, params, 'layer2' + '.0.', self.training, self._modules['layer2']._modules['0']._modules, True)
        x = self.layer3[0].block_forward_para(x, params, 'layer3' + '.0.', self.training, self._modules['layer3']._modules['0']._modules, True)
        x = self.layer4[0].block_forward_para(x, params, 'layer4' + '.0.', self.training, self._modules['layer4']._modules['0']._modules, True)

        if self.keep_avg_pool:
            x = self.avgpool(x)
            
        x = x.view(x.size(0), -1)
        
        if embedding:
            return x
        else:
            # Apply Linear Layer
            logits = F.relu(F.linear(x, weight=params['fc.0.weight'], bias=params['fc.0.bias']))
            logits = F.linear(logits, weight=params['fc.2.weight'], bias=params['fc.2.bias'])
            return logits

class ResNetMAML(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=10):
        self.inplanes = 3
        super(ResNetMAML, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
            
        x = self.layer1[0].block_forward_para(x, params, 'layer1' + '.0.', self.training, self._modules['layer1']._modules['0']._modules, True)
        x = self.layer2[0].block_forward_para(x, params, 'layer2' + '.0.', self.training, self._modules['layer2']._modules['0']._modules, True)
        x = self.layer3[0].block_forward_para(x, params, 'layer3' + '.0.', self.training, self._modules['layer3']._modules['0']._modules, True)
        x = self.layer4[0].block_forward_para(x, params, 'layer4' + '.0.', self.training, self._modules['layer4']._modules['0']._modules, True)

        if self.keep_avg_pool:
            x = self.avgpool(x)
            
        x = x.view(x.size(0), -1)
        
        if embedding:
            return x
        else:
            # Apply Linear Layer
            logits = F.linear(x, weight=params['fc.weight'], bias=params['fc.bias'])
            return logits