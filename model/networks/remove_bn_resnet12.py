"""
Remove BatchNormalization from ResNet12 models.
Multiple approaches: complete removal, replace with Identity, or keep structure.
"""

import torch
import torch.nn as nn


# ============================================================================
# METHOD 1: Remove BatchNorm layers entirely (in-place modification)
# ============================================================================

def remove_batchnorm(model):
    """
    Remove all BatchNorm layers from a model by replacing them with Identity.
    
    Args:
        model: PyTorch model
    
    Returns:
        Modified model without BatchNorm layers
    """
    def remove_bn_recursive(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Replace with Identity (pass-through)
                setattr(module, name, nn.Identity())
                print(f"Removed BatchNorm: {name}")
            else:
                # Recursively process child modules
                remove_bn_recursive(child)
    
    remove_bn_recursive(model)
    return model


# ============================================================================
# METHOD 2: Remove BatchNorm and adjust bias
# ============================================================================

def remove_batchnorm_with_bias_fix(model):
    """
    Remove BatchNorm and add bias to preceding Conv layers.
    This is more principled since BatchNorm includes a bias term.
    
    Args:
        model: PyTorch model
    
    Returns:
        Modified model with Conv bias enabled where BatchNorm was removed
    """
    def process_module(module, parent=None, parent_name=None):
        prev_conv = None
        prev_conv_name = None
        
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                prev_conv = child
                prev_conv_name = name
            
            elif isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Remove BatchNorm
                setattr(module, name, nn.Identity())
                print(f"Removed BatchNorm: {name}")
                
                # Add bias to previous conv if it exists and doesn't have bias
                if prev_conv is not None and prev_conv.bias is None:
                    # Create new conv with bias
                    out_channels = prev_conv.out_channels
                    new_conv = type(prev_conv)(
                        prev_conv.in_channels,
                        prev_conv.out_channels,
                        prev_conv.kernel_size,
                        stride=prev_conv.stride,
                        padding=prev_conv.padding,
                        dilation=prev_conv.dilation,
                        groups=prev_conv.groups,
                        bias=True  # Enable bias
                    )
                    
                    # Copy weights
                    new_conv.weight.data.copy_(prev_conv.weight.data)
                    
                    # Initialize bias with BatchNorm's bias if available
                    if hasattr(child, 'bias') and child.bias is not None:
                        new_conv.bias.data.copy_(child.bias.data)
                    else:
                        new_conv.bias.data.zero_()
                    
                    # Replace the conv
                    setattr(module, prev_conv_name, new_conv)
                    print(f"  Added bias to Conv: {prev_conv_name}")
                
                prev_conv = None
            
            else:
                # Recursively process
                process_module(child, module, name)
    
    process_module(model)
    return model


# ============================================================================
# METHOD 3: Create ResNet12 without BatchNorm from scratch
# ============================================================================

class BasicBlockNoBN(nn.Module):
    """Basic residual block without BatchNorm"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                               padding=1, bias=True)  # bias=True since no BN
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=True)
            )
    
    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNet12NoBN(nn.Module):
    """ResNet12 without any BatchNorm layers"""
    def __init__(self, num_classes=64, channels=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, channels[0], 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        # 4 residual blocks
        self.layer1 = BasicBlockNoBN(channels[0], channels[0], stride=1)
        self.layer2 = BasicBlockNoBN(channels[0], channels[1], stride=2)
        self.layer3 = BasicBlockNoBN(channels[1], channels[2], stride=2)
        self.layer4 = BasicBlockNoBN(channels[2], channels[3], stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================
# METHOD 4: Replace with alternative normalizations
# ============================================================================

class BasicBlockWithLayerNorm(nn.Module):
    """Basic block with LayerNorm instead of BatchNorm"""
    def __init__(self, in_channels, out_channels, stride=1, image_size=84):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, 
                               padding=1, bias=False)
        
        # LayerNorm after first conv
        h, w = image_size // stride, image_size // stride
        self.ln1 = nn.LayerNorm([out_channels, h, w])
        
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # LayerNorm after second conv
        self.ln2 = nn.LayerNorm([out_channels, h, w])
        
        self.relu2 = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.LayerNorm([out_channels, h, w])
            )
    
    def forward(self, x):
        out = self.relu1(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


def replace_batchnorm_with_layernorm(model):
    """
    Replace BatchNorm with LayerNorm.
    Note: LayerNorm requires knowing spatial dimensions.
    """
    def replace_bn_recursive(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.BatchNorm2d,)):
                num_features = child.num_features
                # Create LayerNorm (simplified, assumes square feature maps)
                # In practice, you need to know the exact spatial dimensions
                setattr(module, name, nn.LayerNorm(num_features))
                print(f"Replaced BatchNorm with LayerNorm: {name}")
            else:
                replace_bn_recursive(child)
    
    replace_bn_recursive(model)
    return model


# ============================================================================
# UTILITY: Compare models
# ============================================================================

def compare_models(model_with_bn, model_without_bn):
    """Compare two models parameter-wise"""
    params_with_bn = sum(p.numel() for p in model_with_bn.parameters())
    params_without_bn = sum(p.numel() for p in model_without_bn.parameters())
    
    print(f"Parameters with BatchNorm: {params_with_bn:,}")
    print(f"Parameters without BatchNorm: {params_without_bn:,}")
    print(f"Difference: {params_with_bn - params_without_bn:,}")
    print(f"Reduction: {(1 - params_without_bn/params_with_bn)*100:.2f}%")


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_removal():
    """Demonstrate different methods of removing BatchNorm"""
    
    # First, create a simple model with BatchNorm for testing
    class SimpleResNet12WithBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x).flatten(1)
            return self.fc(x)
    
    print("=" * 80)
    print("METHOD 1: Remove BatchNorm (Replace with Identity)")
    print("=" * 80)
    
    model1 = SimpleResNet12WithBN()
    print("\nOriginal model:")
    print(model1)
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    model1_no_bn = remove_batchnorm(model1)
    print("\nAfter removal:")
    print(model1_no_bn)
    print(f"Parameters: {sum(p.numel() for p in model1_no_bn.parameters()):,}")
    
    # Test
    x = torch.randn(2, 3, 84, 84)
    out = model1_no_bn(x)
    print(f"\nTest output shape: {out.shape}")
    print(f"Contains NaN: {torch.isnan(out).any()}")
    
    print("\n" + "=" * 80)
    print("METHOD 2: Remove BatchNorm and Add Bias to Conv")
    print("=" * 80)
    
    model2 = SimpleResNet12WithBN()
    model2_no_bn = remove_batchnorm_with_bias_fix(model2)
    print("\nAfter removal with bias fix:")
    print(f"Parameters: {sum(p.numel() for p in model2_no_bn.parameters()):,}")
    
    out2 = model2_no_bn(x)
    print(f"\nTest output shape: {out2.shape}")
    print(f"Contains NaN: {torch.isnan(out2).any()}")
    
    print("\n" + "=" * 80)
    print("METHOD 3: Create ResNet12 Without BatchNorm from Scratch")
    print("=" * 80)
    
    model3 = ResNet12NoBN(num_classes=64)
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    out3 = model3(x)
    print(f"\nTest output shape: {out3.shape}")
    print(f"Contains NaN: {torch.isnan(out3).any()}")
    
    # Count layers
    bn_count = sum(1 for m in model1.modules() if isinstance(m, nn.BatchNorm2d))
    conv_count = sum(1 for m in model3.modules() if isinstance(m, nn.Conv2d))
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original model had {bn_count} BatchNorm layers")
    print(f"New model has {conv_count} Conv layers with bias enabled")
    print("\nRecommendations:")
    print("• Method 1: Quick removal, use Identity placeholder")
    print("• Method 2: Better - adds bias to convs for proper learning")
    print("• Method 3: Best - clean implementation from scratch")
    print("• Consider using GroupNorm or LayerNorm instead of removing normalization")
    print("\nWarning: Training without normalization may be unstable!")
    print("Consider using smaller learning rates and weight initialization carefully.")


if __name__ == "__main__":
    demonstrate_removal()
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES:")
    print("=" * 80)
    print("""
# Quick removal (replace with Identity):
model = remove_batchnorm(your_resnet12_model)

# Better removal (add bias to Conv layers):
model = remove_batchnorm_with_bias_fix(your_resnet12_model)

# Create from scratch without BatchNorm:
model = ResNet12NoBN(num_classes=64)

# Training tips without BatchNorm:
# 1. Use smaller learning rate (e.g., 0.001 instead of 0.01)
# 2. Use proper weight initialization (Kaiming/He initialization)
# 3. Consider gradient clipping
# 4. May need more regularization (dropout, weight decay)
    """)
