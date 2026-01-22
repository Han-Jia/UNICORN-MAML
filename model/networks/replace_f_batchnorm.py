"""
Replace F.batch_norm with GroupNorm in models using functional API.
Handles both conversion of existing models and examples of proper implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# METHOD 1: Replace F.batch_norm with F.group_norm
# ============================================================================

def replace_f_batch_norm_in_code():
    """
    Example showing how to replace F.batch_norm with F.group_norm.
    
    Note: F.group_norm requires explicit num_groups parameter.
    """
    
    # BEFORE: Using F.batch_norm
    class ModelWithFBatchNorm(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Conv2d(3, channels, 3, padding=1)
            # BatchNorm parameters
            self.bn_weight = nn.Parameter(torch.ones(channels))
            self.bn_bias = nn.Parameter(torch.zeros(channels))
            self.register_buffer('running_mean', torch.zeros(channels))
            self.register_buffer('running_var', torch.ones(channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        def forward(self, x):
            x = self.conv(x)
            # Using F.batch_norm
            x = F.batch_norm(
                x, 
                self.running_mean, 
                self.running_var,
                self.bn_weight, 
                self.bn_bias,
                training=self.training,
                momentum=0.1,
                eps=1e-5
            )
            return x
    
    # AFTER: Using F.group_norm
    class ModelWithFGroupNorm(nn.Module):
        def __init__(self, channels, num_groups=8):
            super().__init__()
            self.conv = nn.Conv2d(3, channels, 3, padding=1)
            self.num_groups = num_groups
            # GroupNorm parameters
            self.gn_weight = nn.Parameter(torch.ones(channels))
            self.gn_bias = nn.Parameter(torch.zeros(channels))
        
        def forward(self, x):
            x = self.conv(x)
            # Using F.group_norm (no running statistics needed!)
            x = F.group_norm(
                x,
                self.num_groups,
                self.gn_weight,
                self.gn_bias,
                eps=1e-5
            )
            return x
    
    return ModelWithFBatchNorm, ModelWithFGroupNorm


# ============================================================================
# METHOD 2: Convert existing model with F.batch_norm to use nn.GroupNorm
# ============================================================================

def convert_f_batchnorm_to_groupnorm_layer(model, num_groups=8):
    """
    Strategy: Replace F.batch_norm calls with nn.GroupNorm layers.
    
    This requires modifying the model's forward() method, which can be done by:
    1. Subclassing and overriding forward()
    2. Monkey-patching the forward method
    3. Creating a wrapper
    
    Here's the wrapper approach:
    """
    
    class GroupNormWrapper(nn.Module):
        def __init__(self, original_model, num_groups=8):
            super().__init__()
            self.model = original_model
            self.group_norms = nn.ModuleDict()
            
            # Find all parameters that look like batch norm params
            # and create corresponding GroupNorm layers
            for name, param in original_model.named_parameters():
                if 'bn_weight' in name or 'weight' in name:
                    # Extract the base name
                    base_name = name.replace('.weight', '').replace('bn_weight', 'bn')
                    
                    # Check if there's a corresponding bias
                    bias_name = name.replace('weight', 'bias').replace('bn_weight', 'bn_bias')
                    
                    if hasattr(original_model, bias_name.split('.')[-1]):
                        num_channels = param.shape[0]
                        groups = min(num_groups, num_channels)
                        
                        # Adjust groups to be divisor of channels
                        while num_channels % groups != 0 and groups > 1:
                            groups -= 1
                        
                        gn = nn.GroupNorm(groups, num_channels, eps=1e-5, affine=True)
                        
                        # Copy weights
                        gn.weight.data.copy_(param.data)
                        bias_param = getattr(original_model, bias_name.split('.')[-1])
                        gn.bias.data.copy_(bias_param.data)
                        
                        self.group_norms[base_name] = gn
        
        def forward(self, x):
            # This needs to be customized based on the original model
            # Here's a general template
            return self.model(x)
    
    return GroupNormWrapper(model, num_groups)


# ============================================================================
# METHOD 3: Direct replacement function for F.batch_norm calls
# ============================================================================

def create_group_norm_function(num_groups=8):
    """
    Create a drop-in replacement function for F.batch_norm.
    
    Usage: Replace F.batch_norm calls with this function.
    """
    
    def group_norm_replacement(input, running_mean, running_var, weight=None, 
                              bias=None, training=False, momentum=0.1, eps=1e-5):
        """
        Drop-in replacement for F.batch_norm that uses group normalization.
        
        Args:
            input: Input tensor (N, C, H, W) or (N, C, L)
            running_mean: Ignored (kept for API compatibility)
            running_var: Ignored (kept for API compatibility)
            weight: Learnable weight (gamma)
            bias: Learnable bias (beta)
            training: Ignored (GroupNorm behaves same in train/eval)
            momentum: Ignored (no running stats in GroupNorm)
            eps: Small constant for numerical stability
        
        Returns:
            Normalized tensor
        """
        num_channels = input.shape[1]
        groups = min(num_groups, num_channels)
        
        # Ensure groups divides channels
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        
        return F.group_norm(input, groups, weight, bias, eps)
    
    return group_norm_replacement


# ============================================================================
# COMPLETE EXAMPLE: ResNet-style Block
# ============================================================================

class ResNetBlockWithFBatchNorm(nn.Module):
    """Original block using F.batch_norm"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # BatchNorm parameters
        self.bn1_weight = nn.Parameter(torch.ones(out_channels))
        self.bn1_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('bn1_running_mean', torch.zeros(out_channels))
        self.register_buffer('bn1_running_var', torch.ones(out_channels))
        
        self.bn2_weight = nn.Parameter(torch.ones(out_channels))
        self.bn2_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer('bn2_running_mean', torch.zeros(out_channels))
        self.register_buffer('bn2_running_var', torch.ones(out_channels))
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.batch_norm(out, self.bn1_running_mean, self.bn1_running_var,
                          self.bn1_weight, self.bn1_bias, 
                          training=self.training, momentum=0.1, eps=1e-5)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = F.batch_norm(out, self.bn2_running_mean, self.bn2_running_var,
                          self.bn2_weight, self.bn2_bias,
                          training=self.training, momentum=0.1, eps=1e-5)
        out = self.relu(out)
        return out


class ResNetBlockWithFGroupNorm(nn.Module):
    """Converted block using F.group_norm"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        self.num_groups = num_groups
        
        # GroupNorm parameters (no running statistics!)
        self.gn1_weight = nn.Parameter(torch.ones(out_channels))
        self.gn1_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.gn2_weight = nn.Parameter(torch.ones(out_channels))
        self.gn2_bias = nn.Parameter(torch.zeros(out_channels))
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.group_norm(out, self.num_groups, 
                          self.gn1_weight, self.gn1_bias, eps=1e-5)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = F.group_norm(out, self.num_groups,
                          self.gn2_weight, self.gn2_bias, eps=1e-5)
        out = self.relu(out)
        return out


class ResNetBlockWithNNGroupNorm(nn.Module):
    """Best practice: using nn.GroupNorm layers"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.relu(self.gn2(self.conv2(out)))
        return out


# ============================================================================
# AUTOMATIC CONVERSION UTILITY
# ============================================================================

def auto_convert_f_batchnorm(model_class, num_groups=8):
    """
    Automatically convert a model class using F.batch_norm to use F.group_norm.
    
    This modifies the forward method to intercept F.batch_norm calls.
    """
    import types
    
    original_forward = model_class.forward
    
    def new_forward(self, x):
        # Monkey-patch F.batch_norm temporarily
        original_batch_norm = F.batch_norm
        
        def group_norm_replacement(input, running_mean, running_var, weight=None,
                                  bias=None, training=False, momentum=0.1, eps=1e-5):
            num_channels = input.shape[1]
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            return F.group_norm(input, groups, weight, bias, eps)
        
        F.batch_norm = group_norm_replacement
        result = original_forward(self, x)
        F.batch_norm = original_batch_norm
        
        return result
    
    model_class.forward = new_forward
    return model_class


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_conversion():
    print("=" * 80)
    print("COMPARING F.batch_norm vs F.group_norm")
    print("=" * 80)
    
    # Create models
    model_bn = ResNetBlockWithFBatchNorm(64, 128)
    model_gn = ResNetBlockWithFGroupNorm(64, 128, num_groups=8)
    model_nn = ResNetBlockWithNNGroupNorm(64, 128, num_groups=8)
    
    # Test with small batch
    batch_size = 2
    x = torch.randn(batch_size, 64, 32, 32)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Batch size: {batch_size} (very small)")
    
    # Test BatchNorm version
    model_bn.eval()
    with torch.no_grad():
        out_bn = model_bn(x)
    print(f"\nF.batch_norm output shape: {out_bn.shape}")
    print(f"Contains NaN: {torch.isnan(out_bn).any()}")
    print(f"Parameters: {sum(p.numel() for p in model_bn.parameters()):,}")
    
    # Test GroupNorm version
    model_gn.eval()
    with torch.no_grad():
        out_gn = model_gn(x)
    print(f"\nF.group_norm output shape: {out_gn.shape}")
    print(f"Contains NaN: {torch.isnan(out_gn).any()}")
    print(f"Parameters: {sum(p.numel() for p in model_gn.parameters()):,}")
    
    # Test nn.GroupNorm version
    model_nn.eval()
    with torch.no_grad():
        out_nn = model_nn(x)
    print(f"\nnn.GroupNorm output shape: {out_nn.shape}")
    print(f"Contains NaN: {torch.isnan(out_nn).any()}")
    print(f"Parameters: {sum(p.numel() for p in model_nn.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES:")
    print("=" * 80)
    print("F.batch_norm:")
    print("  - Requires running_mean and running_var buffers")
    print("  - Different behavior in train vs eval mode")
    print("  - Unstable with small batches")
    print("\nF.group_norm:")
    print("  - No running statistics needed")
    print("  - Same behavior in train and eval mode")
    print("  - Stable with any batch size")
    print("\nnn.GroupNorm (RECOMMENDED):")
    print("  - Cleaner API, easier to use")
    print("  - Parameters managed automatically")
    print("  - Same benefits as F.group_norm")


if __name__ == "__main__":
    demonstrate_conversion()
    
    print("\n" + "=" * 80)
    print("QUICK REFERENCE:")
    print("=" * 80)
    print("""
# Replace this:
x = F.batch_norm(x, running_mean, running_var, weight, bias, 
                 training=self.training, momentum=0.1, eps=1e-5)

# With this:
x = F.group_norm(x, num_groups=8, weight, bias, eps=1e-5)

# Or better yet, use nn.GroupNorm:
self.gn = nn.GroupNorm(8, num_channels)
x = self.gn(x)
    """)
