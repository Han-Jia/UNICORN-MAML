"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair

__all__ = ['SKConv2d']

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size, share_channel=False):
        super(DropBlock2D, self).__init__()
        self.register_buffer('i', torch.zeros(1, dtype=torch.int64))
        self.register_buffer('drop_prob', drop_prob * torch.ones(1, dtype=torch.float32))
        self.inited = False
        self.step_size = 0.0
        self.start_step = 0
        self.nr_steps = 0
        self.block_size = block_size
        self.share_channel = share_channel

    def reset(self):
        """stop DropBlock"""
        self.inited = True
        self.i[0] = 0
        self.drop_prob = 0.0

    def reset_steps(self, start_step, nr_steps, start_value=0, stop_value=None):
        self.inited = True
        stop_value = self.drop_prob.item() if stop_value is None else stop_value
        self.i[0] = 0
        self.drop_prob[0] = start_value
        self.step_size = (stop_value - start_value) / nr_steps
        self.nr_steps = nr_steps
        self.start_step = start_step

    def forward(self, x):
        if not self.training or self.drop_prob.item() == 0.:
            return x
        else:
            self.step()

            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask and place on input device
            if self.share_channel:
                mask = (torch.rand(x.shape[0], *x.shape[2:], device=x.device, dtype=x.dtype) < gamma).squeeze(1)
            else:
                mask = (torch.rand(*x.shape, device=x.device, dtype=x.dtype) < gamma)

            # compute block mask
            block_mask, keeped = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask

            # scale output
            out = out * (block_mask.numel() / keeped).to(out)
            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask,
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        keeped = block_mask.numel() - block_mask.sum().to(torch.float32)
        block_mask = 1 - block_mask

        return block_mask, keeped

    def _compute_gamma(self, x):
        _, c, h, w = x.size()
        gamma = self.drop_prob.item() / (self.block_size ** 2) * (h * w) / \
            ((w - self.block_size + 1) * (h - self.block_size + 1))
        return gamma

    def step(self):
        assert self.inited
        idx = self.i.item()
        if idx > self.start_step and idx < self.start_step + self.nr_steps:
            self.drop_prob += self.step_size
        self.i += 1

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        idx_key = prefix + 'i'
        drop_prob_key = prefix + 'drop_prob'
        if idx_key not in state_dict:
            state_dict[idx_key] =  torch.zeros(1, dtype=torch.int64)
        if idx_key not in drop_prob_key:
            state_dict[drop_prob_key] =  torch.ones(1, dtype=torch.float32)
        super(DropBlock2D, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """overwrite save method"""
        pass

    def extra_repr(self):
        return 'drop_prob={}, step_size={}'.format(self.drop_prob, self.step_size)

def reset_dropblock(start_step, nr_steps, start_value, stop_value, m):
    """
    Example:
        from functools import partial
        apply_drop_prob = partial(reset_dropblock, 0, epochs*iters_per_epoch, 0.0, 0.1)
        net.apply(apply_drop_prob)
    """
    if isinstance(m, DropBlock2D):
        print('reseting dropblock')
        m.reset_steps(start_step, nr_steps, start_value, stop_value)

class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                           groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, channel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, channel//self.radix, dim=1)
            gap = sum(splited) 
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        if self.radix > 1:
            atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        else:
            atten = F.sigmoid(atten, dim=1).view(batch, -1, 1, 1)

        if self.radix > 1:
            atten = torch.split(atten, channel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(atten, splited)])
        else:
            out = atten * x
        return out.contiguous()
