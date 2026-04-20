"""
Self-Pruning Neural Network with Pretrained ResNet-18

This module implements a prunable ResNet-18 using pretrained weights.
Approach:
1. Load pretrained ResNet-18 (trained on ImageNet)
2. Modify final FC layer to use PrunableLinear
3. Add a bottleneck layer before the final classifier
4. Freeze most backbone layers

Key Concepts:
- Transfer learning: Use pretrained features
- Pruning: Learnable gates on final layers
- Fine-tuning: Minimal training on CIFAR-10
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class PrunableLinear(nn.Module):
    """
    A linear layer with learnable pruning gates.
    
    Gates use sigmoid activation (0 to 1) to softly mask weights.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Gate parameters - one gate per weight
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)
    
    def get_gates(self) -> torch.Tensor:
        """Return current gate values after sigmoid."""
        return torch.sigmoid(self.gate_scores)
    
    def get_gate_l1_loss(self) -> torch.Tensor:
        """Return L1 loss (sum of all gate values)."""
        return torch.sigmoid(self.gate_scores).sum()


class PrunableResNet18(nn.Module):
    """
    ResNet-18 with self-pruning final layers for CIFAR-10.
    
    Architecture modifications:
    - Load pretrained ResNet-18 backbone
    - Replace avgpool with adaptive pooling
    - Add bottleneck: 512 -> 256
    - Add prunable classifier: 256 -> 10
    
    Only bottleneck + classifier are trained (with gates).
    Backbone is frozen.
    """
    
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Extract backbone (conv layers only, before final FC)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bottleneck layer (trainable)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prunable final classifier
        self.classifier = PrunableLinear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features (frozen)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Bottleneck (trainable)
        x = self.bottleneck(x)
        
        # Prunable classifier
        x = self.classifier(x)
        
        return x
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """L1 sparsity loss from classifier gates."""
        return self.classifier.get_gate_l1_loss()
    
    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Compute sparsity as % of gates below threshold.
        """
        gates = self.classifier.get_gates().flatten()
        num_pruned = (gates < threshold).sum().item()
        return (num_pruned / gates.numel()) * 100
    
    def get_all_gates(self) -> torch.Tensor:
        """Return all gate values for visualization."""
        return self.classifier.get_gates().flatten()
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
