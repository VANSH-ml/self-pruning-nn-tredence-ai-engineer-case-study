"""
Training script for Self-Pruning ResNet-18

Uses transfer learning with minimal fine-tuning:
- 1-3 epochs only
- Small learning rate
- Only bottleneck + classifier layers trained
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from model_resnet import PrunableResNet18
from utils import get_cifar10_loaders


def train_epoch_resnet(model: nn.Module, train_loader, optimizer, 
                       lambda_sparsity: float, device: torch.device) -> Dict[str, float]:
    """
    Train one epoch with sparsity loss.
    
    Loss = CrossEntropy + λ * sum(gates)
    """
    model.train()
    
    # Backbone stays in eval mode (frozen)
    model.backbone.eval()
    
    total_loss = 0.0
    ce_loss_total = 0.0
    sparsity_loss_total = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Losses
        ce_loss = criterion(outputs, labels)
        sparsity_loss = model.get_sparsity_loss()
        loss = ce_loss + lambda_sparsity * sparsity_loss
        
        # Backward (only trainable params get gradients)
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item() * images.size(0)
        ce_loss_total += ce_loss.item() * images.size(0)
        sparsity_loss_total += sparsity_loss.item() * images.size(0)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return {
        'loss': total_loss / total,
        'ce_loss': ce_loss_total / total,
        'sparsity_loss': sparsity_loss_total / total,
        'accuracy': 100. * correct / total
    }


def evaluate_resnet(model: nn.Module, test_loader, device: torch.device) -> tuple:
    """
    Evaluate model on test set.
    
    Returns:
        (accuracy, sparsity)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    sparsity = model.compute_sparsity(threshold=1e-2)
    
    return accuracy, sparsity


def train_resnet(lambda_sparsity: float, num_epochs: int = 2,
                 lr: float = 1e-4, device: torch.device = None) -> Dict:
    """
    Train prunable ResNet-18 with transfer learning.
    
    Args:
        lambda_sparsity: Weight for L1 sparsity loss
        num_epochs: Training epochs (1-3 recommended)
        lr: Small learning rate for fine-tuning
        device: GPU/CPU
    
    Returns:
        Dictionary with results and trained model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Training ResNet-18 with λ = {lambda_sparsity:.0e}")
    print(f"Epochs: {num_epochs}, LR: {lr}")
    print(f"Device: {device}")
    print(f"{'='*70}")
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=0)
    
    # Initialize model with pretrained weights
    model = PrunableResNet18(num_classes=10, freeze_backbone=True)
    model = model.to(device)
    
    # Statistics
    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
    
    # Optimizer - only trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Training
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'sparsity': []}
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_resnet(model, train_loader, optimizer, 
                                            lambda_sparsity, device)
        test_acc, sparsity = evaluate_resnet(model, test_loader, device)
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_acc'].append(test_acc)
        history['sparsity'].append(sparsity)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Sparsity: {sparsity:.2f}%")
    
    # Final evaluation
    final_acc, final_sparsity = evaluate_resnet(model, test_loader, device)
    
    print(f"\nFinal Results (λ = {lambda_sparsity:.0e}):")
    print(f"  Test Accuracy: {final_acc:.2f}%")
    print(f"  Sparsity: {final_sparsity:.2f}%")
    
    return {
        'lambda': lambda_sparsity,
        'model': model,
        'accuracy': final_acc,
        'sparsity': final_sparsity,
        'history': history,
        'trainable_params': trainable_params
    }


def run_resnet_experiments(lambdas: List[float] = None, num_epochs: int = 2) -> List[Dict]:
    """
    Run experiments with different lambda values.
    
    Default: [1e-4, 1e-3, 1e-2]
    """
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = []
    for lam in lambdas:
        result = train_resnet(lam, num_epochs=num_epochs, device=device)
        results.append(result)
    
    return results
