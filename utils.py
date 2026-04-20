"""
Self-Pruning Neural Network - Utilities

This module contains utility functions for training, evaluation, and visualization.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with standard preprocessing.
    
    CIFAR-10: 60,000 32x32 color images in 10 classes
    - 50,000 training images
    - 10,000 test images
    """
    # Standard normalization for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                lambda_sparsity: float, device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Total Loss = CrossEntropyLoss + lambda * SparsityLoss
    
    Sparsity Loss: L1 norm of gates (sum of all gate values)
    - Encourages gates to shrink toward 0
    - Creates sparsity by pushing many gates below threshold
    """
    model.train()
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
        
        # Classification loss
        ce_loss = criterion(outputs, labels)
        
        # Sparsity loss: L1 on gates (encourages gates -> 0)
        sparsity_loss = model.get_total_sparsity_loss()
        
        # Total loss with sparsity regularization
        loss = ce_loss + lambda_sparsity * sparsity_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
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


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        accuracy: Test accuracy percentage
        sparsity: Percentage of gates below threshold (1e-2)
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


def plot_gate_histogram(model: nn.Module, lambda_val: float, save_path: str = None):
    """
    Plot histogram of gate values.
    
    With L1 regularization, gates should cluster near 0 (pruned) or 1 (active).
    """
    gates = model.get_all_gates().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.hist(gates, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Gate Value (sigmoid(gate_scores))')
    plt.ylabel('Count')
    plt.title(f'Gate Value Distribution (λ = {lambda_val})')
    plt.axvline(x=1e-2, color='red', linestyle='--', label=f'Threshold (1e-2)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_results(results: List[Dict], save_path: str = None):
    """
    Plot Accuracy vs Lambda and Sparsity vs Lambda.
    
    Shows the trade-off: higher lambda → more sparsity but potentially lower accuracy.
    """
    lambdas = [r['lambda'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    sparsities = [r['sparsity'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs Lambda
    ax1.plot(lambdas, accuracies, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xscale('log')
    ax1.set_xlabel('Lambda (log scale)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Sparsity Regularization', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Sparsity vs Lambda
    ax2.plot(lambdas, sparsities, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xscale('log')
    ax2.set_xlabel('Lambda (log scale)', fontsize=12)
    ax2.set_ylabel('Sparsity (%)', fontsize=12)
    ax2.set_title('Sparsity vs Regularization Strength', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def print_results_table(results: List[Dict]):
    """Print results in formatted table."""
    print("\n" + "=" * 50)
    print("RESULTS TABLE")
    print("=" * 50)
    print(f"{'Lambda':<15} | {'Accuracy':<12} | {'Sparsity':<12}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['lambda']:<15.0e} | {r['accuracy']:<12.2f} | {r['sparsity']:<12.2f}")
    
    print("=" * 50)
