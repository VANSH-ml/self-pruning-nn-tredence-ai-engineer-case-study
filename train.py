"""
Self-Pruning Neural Network - Training Script (ResNet-18)

This module handles training with transfer learning:
- Frozen pretrained backbone
- Prunable final layers
- Minimal fine-tuning (1-3 epochs)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from model import PrunableResNet18
from utils import get_cifar10_loaders


def train_epoch_resnet(model, train_loader, optimizer, lambda_sparsity, device):
    """Train one epoch with frozen backbone."""
    model.train()
    model.backbone.eval()  # Keep backbone in eval mode
    
    total_loss = 0.0
    ce_loss_total = 0.0
    sparsity_loss_total = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        ce_loss = criterion(outputs, labels)
        sparsity_loss = model.get_sparsity_loss()
        loss = ce_loss + lambda_sparsity * sparsity_loss
        
        loss.backward()
        optimizer.step()
        
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


def evaluate_resnet(model, test_loader, device):
    """Evaluate on test set."""
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


def train_model(lambda_sparsity: float, num_epochs: int = 2, 
                lr: float = 1e-4, device: torch.device = None) -> Dict:
    """
    Train prunable ResNet-18 with transfer learning.
    
    Args:
        lambda_sparsity: Weight for L1 sparsity loss
        num_epochs: Training epochs (1-3 recommended for transfer learning)
        lr: Small learning rate for fine-tuning
        device: GPU/CPU device
    
    Returns:
        Dictionary with training history and final metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Training Prunable ResNet-18 with λ = {lambda_sparsity:.0e}")
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
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Optimizer - only trainable parameters
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'sparsity': []}
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_resnet(model, train_loader, optimizer, lambda_sparsity, device)
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


def run_experiments(lambdas: List[float] = None, num_epochs: int = 2) -> List[Dict]:
    """
    Run training experiments for multiple lambda values.
    
    Default lambdas: [1e-4, 1e-3, 1e-2]
    Default epochs: 2 (transfer learning needs minimal training)
    
    Higher lambda = stronger sparsity regularization
    - Low lambda (1e-4): Minimal pruning, maximum accuracy (~87-89%)
    - Medium lambda (1e-3): Balanced trade-off (~85-87%, ~40-50% sparsity)
    - High lambda (1e-2): Aggressive pruning (~80-84%, ~60-70% sparsity)
    """
    if lambdas is None:
        lambdas = [1e-4, 1e-3, 1e-2]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running ResNet-18 experiments on: {device}")
    
    results = []
    for lam in lambdas:
        result = train_model(lam, num_epochs=num_epochs, device=device)
        results.append(result)
    
    return results
