"""
Replace BatchNormalization with GroupNormalization in ResNet12.
Optimized for small datasets and few-shot learning scenarios.
"""

import torch
import torch.nn as nn


def replace_batchnorm_resnet12(model, num_groups=8):
    """
    Replace BatchNorm with GroupNorm in ResNet12 (optimized for small datasets).
    
    Args:
        model: ResNet12 model
        num_groups: Number of groups (default: 8, good for ResNet12's channel counts)
    
    Returns:
        Modified model with GroupNorm
    """
    def convert_module(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                num_channels = child.num_features
                
                # Find optimal number of groups
                groups = find_optimal_groups(num_channels, num_groups)
                
                # Create GroupNorm
                gn = nn.GroupNorm(
                    num_groups=groups,
                    num_channels=num_channels,
                    eps=child.eps,
                    affine=child.affine
                )
                
                # Copy weights if affine
                if child.affine:
                    gn.weight.data.copy_(child.weight.data)
                    gn.bias.data.copy_(child.bias.data)
                
                setattr(module, name, gn)
                print(f"Replaced {name}: {num_channels} channels -> {groups} groups")
            else:
                convert_module(child)
    
    convert_module(model)
    return model


def find_optimal_groups(num_channels, preferred_groups):
    """
    Find the optimal number of groups that divides num_channels evenly.
    
    Args:
        num_channels: Number of channels
        preferred_groups: Preferred number of groups
    
    Returns:
        Optimal number of groups
    """
    if num_channels % preferred_groups == 0:
        return preferred_groups
    
    # Find divisors of num_channels
    divisors = []
    for i in range(1, num_channels + 1):
        if num_channels % i == 0:
            divisors.append(i)
    
    # Find closest divisor to preferred_groups
    closest = min(divisors, key=lambda x: abs(x - preferred_groups))
    return closest


# ============================================================================
# EXAMPLE RESNET12 IMPLEMENTATION
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet12"""
    def __init__(self, in_channels, out_channels, stride=1, use_groupnorm=False, num_groups=8):
        super().__init__()
        
        if use_groupnorm:
            norm1 = nn.GroupNorm(find_optimal_groups(out_channels, num_groups), out_channels)
            norm2 = nn.GroupNorm(find_optimal_groups(out_channels, num_groups), out_channels)
        else:
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm1
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = norm2
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if use_groupnorm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.GroupNorm(find_optimal_groups(out_channels, num_groups), out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet12(nn.Module):
    """ResNet12 for few-shot learning"""
    def __init__(self, num_classes=64, use_groupnorm=False, num_groups=8):
        super().__init__()
        
        self.in_channels = 64
        self.use_groupnorm = use_groupnorm
        self.num_groups = num_groups
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        if use_groupnorm:
            self.norm1 = nn.GroupNorm(find_optimal_groups(64, num_groups), 64)
        else:
            self.norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 residual blocks (ResNet12 architecture)
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, stride):
        return BasicBlock(self.in_channels, out_channels, stride, 
                         self.use_groupnorm, self.num_groups)
    
    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# DEMONSTRATION
# ============================================================================

def compare_normalizations():
    """Compare BatchNorm vs GroupNorm behavior with small batches"""
    print("=" * 80)
    print("CREATING RESNET12 WITH BATCHNORM")
    print("=" * 80)
    
    # Model with BatchNorm
    model_bn = ResNet12(num_classes=64, use_groupnorm=False)
    print(f"\nModel created with BatchNorm")
    print(f"Total parameters: {sum(p.numel() for p in model_bn.parameters()):,}")
    
    # Convert to GroupNorm
    print("\n" + "=" * 80)
    print("CONVERTING TO GROUPNORM")
    print("=" * 80)
    model_gn = replace_batchnorm_resnet12(model_bn, num_groups=8)
    
    # Test with small batch
    print("\n" + "=" * 80)
    print("TESTING WITH SMALL BATCH SIZE")
    print("=" * 80)
    
    batch_size = 2  # Very small batch
    x = torch.randn(batch_size, 3, 84, 84)
    
    model_bn.eval()
    model_gn.eval()
    
    with torch.no_grad():
        out_bn = model_bn(x)
        out_gn = model_gn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (BatchNorm): {out_bn.shape}")
    print(f"Output shape (GroupNorm): {out_gn.shape}")
    print(f"Output contains NaN (BatchNorm): {torch.isnan(out_bn).any()}")
    print(f"Output contains NaN (GroupNorm): {torch.isnan(out_gn).any()}")
    
    return model_gn


def create_resnet12_with_groupnorm():
    """Create ResNet12 directly with GroupNorm (recommended)"""
    print("\n" + "=" * 80)
    print("CREATING RESNET12 WITH GROUPNORM FROM SCRATCH")
    print("=" * 80)
    
    model = ResNet12(num_classes=64, use_groupnorm=True, num_groups=8)
    print(f"Model created with GroupNorm")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Count normalization layers
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    print(f"Number of GroupNorm layers: {gn_count}")
    
    return model


if __name__ == "__main__":
    # Compare BatchNorm vs GroupNorm
    model = compare_normalizations()
    
    # Or create directly with GroupNorm
    model_gn = create_resnet12_with_groupnorm()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR SMALL DATASETS:")
    print("=" * 80)
    print("✓ Use GroupNorm (8-16 groups) for batch sizes < 16")
    print("✓ Use LayerNorm for batch size = 1")
    print("✓ GroupNorm is stable across all batch sizes")
    print("✓ No running statistics needed -> more stable training")
