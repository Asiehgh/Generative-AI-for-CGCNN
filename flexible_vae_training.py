#!/usr/bin/env python3
"""
Flexible VAE Training Script that automatically adapts to datasets with or without extra features
Command-line interface for training Crystal Graph VAE on any dataset
"""

import argparse
import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from data import collate_pool
from balanced_vae_model import BalancedCrystalGraphVAE, balanced_vae_loss_function


# Import flexible modules
from flexible_vae_model import (
    FlexibleCrystalGraphVAE, 
    FlexibleVAETrainer, 
    create_flexible_vae_model,
    beta_annealing_schedule
)
from flexible_data_loader import (
    create_flexible_dataset, 
    flexible_collate_pool, 
    get_dataset_info
)
from data import get_train_val_test_loader

def main():
    parser = argparse.ArgumentParser(description='Flexible Crystal Graph VAE Training')
    
    # Data arguments
    parser.add_argument('data_dir', type=str,
                        help='Path to root directory containing CIF files and id_prop.csv')
    parser.add_argument('--feature-file', type=str, default=None,
                        help='Path to extra features CSV file (optional)')
    parser.add_argument('--force-structure-only', action='store_true',
                        help='Force structure-only mode even if feature file exists')
    
    # Model arguments
    parser.add_argument('--latent-dim', default=64, type=int,
                        help='Dimension of latent space (default: 64)')
    parser.add_argument('--atom-fea-len', default=128, type=int,
                        help='Number of hidden atom features in conv layers')
    parser.add_argument('--h-fea-len', default=256, type=int,
                        help='Number of hidden features after pooling')
    parser.add_argument('--n-conv', default=3, type=int,
                        help='Number of conv layers')
    parser.add_argument('--max-atoms', default=200, type=int,
                        help='Maximum number of atoms to generate')
    
    # Training arguments
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--weight-decay', default=1e-5, type=float,
                        help='Weight decay')
    
    # VAE specific arguments
    parser.add_argument('--beta-max', default=1.0, type=float,
                        help='Maximum beta value for beta-VAE')
    parser.add_argument('--beta-annealing-epochs', default=50, type=int,
                        help='Number of epochs for beta annealing')
    parser.add_argument('--warmup-epochs', default=10, type=int,
                        help='Number of warmup epochs')
    
    # Data split arguments
    parser.add_argument('--train-ratio', default=0.8, type=float,
                        help='Training data ratio')
    parser.add_argument('--val-ratio', default=0.1, type=float,
                        help='Validation data ratio')
    parser.add_argument('--test-ratio', default=0.1, type=float,
                        help='Test data ratio')
    
    # System arguments
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./flexible_vae_checkpoints',
                        help='Directory to save model checkpoints')
    
    parser.add_argument('--structure-weight', default=1.0, type=float,
                        help='Weight for structure reconstruction loss')
    parser.add_argument('--feature-weight', default=1.0, type=float,
                        help='Weight for feature reconstruction loss')
    parser.add_argument('--balanced-training', action='store_true',
                        help='Use balanced structure + feature training')
    
    # Generation arguments
    parser.add_argument('--generate-samples', default=0, type=int,
                        help='Number of samples to generate after training')
    
    args = parser.parse_args()
    
    # Setup device
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Auto-detect dataset type and create appropriate dataset
    print("="*60)
    print("DATASET DETECTION AND SETUP")
    print("="*60)
    
    dataset = create_flexible_dataset(
        root_dir=args.data_dir,
        feature_file=args.feature_file,
        force_structure_only=args.force_structure_only
    )
    
    # Get dataset information
    dataset_info = get_dataset_info(dataset)
    print(f"\nDataset Information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Create data loaders with flexible collate function
    print(f"\nCreating data loaders...")
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        pin_memory=args.cuda,
        return_test=True,
        train_size=None,  # Add these missing parameters
        val_size=None,
        test_size=None
    )

    print(f"Data splits:")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create flexible VAE model
    print("\n" + "="*60)
    print("MODEL CREATION")
    print("="*60)
  

    if args.balanced_training:
        print("Using Balanced CGCNN + Features VAE")
        model = BalancedCrystalGraphVAE(
            orig_atom_fea_len=dataset_info['atom_feature_dim'],
            nbr_fea_len=dataset_info['bond_feature_dim'],
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            h_fea_len=args.h_fea_len,
            n_extra_features=dataset_info['num_extra_features'],
            latent_dim=args.latent_dim,
            max_atoms=200,
            use_extra_features=dataset_info['has_extra_features'],
            structure_weight=args.structure_weight,
            feature_weight=args.feature_weight
        )
    else:
        print("Using Original Flexible VAE")
        model = create_flexible_vae_model(
            dataset=dataset,
            latent_dim=args.latent_dim,
            device=device,
            atom_fea_len=args.atom_fea_len,
            n_conv=args.n_conv,
            h_fea_len=args.h_fea_len
        )
   
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Uses extra features: {model.use_extra_features}")
    print(f"  Latent dimension: {model.latent_dim}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Beta annealing schedule
    def beta_schedule(epoch):
        return beta_annealing_schedule(epoch, args.beta_max, args.beta_annealing_epochs)
    
    # Create trainer
    trainer = FlexibleVAETrainer(
        model=model, 
        optimizer=optimizer, 
        device=device, 
        beta_schedule=beta_schedule,
        use_extra_features=model.use_extra_features
    )
    if args.balanced_training:
        trainer.balanced_training = True
        trainer.structure_weight = args.structure_weight
        trainer.feature_weight = args.feature_weight
   
    # Training configuration summary
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Optimizer: Adam")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Beta Annealing: 0 → {args.beta_max} over {args.beta_annealing_epochs} epochs")
    print(f"Dataset Mode: {'With Extra Features' if model.use_extra_features else 'Structure Only'}")
    if model.use_extra_features:
        print(f"Number of Extra Features: {model.n_extra_features}")
    print("="*60)
    
    # Training loop
    print("\nStarting flexible VAE training...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print("-" * 40)
        
        # Training
        train_loss_dict = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kl_loss = 0
        val_extra_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # Handle different batch formats
                inputs, targets, cif_ids = batch_data
                
                if len(inputs) == 5:  # Has extra features
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea = inputs
                else:  # Structure only
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                    extra_fea = None
                
                # Move to device
                if args.cuda:
                    atom_fea = atom_fea.cuda()
                    nbr_fea = nbr_fea.cuda()
                    nbr_fea_idx = nbr_fea_idx.cuda()
                    crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
                    if extra_fea is not None:
                        extra_fea = extra_fea.cuda()
                
                # Forward pass
                output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
                
                # Calculate validation loss
                if model.use_extra_features and extra_fea is not None:
                    recon_loss = torch.nn.functional.mse_loss(
                        output['reconstructed']['extra_features'], extra_fea
                    )
                    val_extra_loss += recon_loss.item()
                else:
                    recon_loss = torch.tensor(0.0)
                
                kl_loss = -0.5 * torch.sum(
                    1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp()
                )
                
                beta = beta_schedule(epoch)
                total_loss = recon_loss + beta * kl_loss
                
                val_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        # Average validation losses
        val_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)
        if model.use_extra_features:
            val_extra_loss /= len(val_loader)
        
        # Record losses
        train_losses.append(train_loss_dict['total_loss'])
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_dict['total_loss'],
                'val_loss': val_loss,
                'latent_dim': args.latent_dim,
                'use_extra_features': model.use_extra_features,
                'n_extra_features': model.n_extra_features,
                'feature_names': getattr(dataset, 'feature_names', []),
                'dataset_info': dataset_info
            }, os.path.join(args.save_dir, 'best_flexible_vae_model.pth'))
            print(f"🎯 New best validation loss: {val_loss:.4f}")
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss_dict['total_loss']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Recon: {val_recon_loss:.4f}")
        print(f"  Val KL: {val_kl_loss:.4f}")
        if model.use_extra_features:
            print(f"  Val Extra Features: {val_extra_loss:.4f}")
        print(f"  Beta: {beta_schedule(epoch):.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save regular checkpoint
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'use_extra_features': model.use_extra_features
            }, os.path.join(args.save_dir, f'flexible_vae_checkpoint_epoch_{epoch+1}.pth'))
    
    # Training completion
    total_training_time = time.time() - training_start_time
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    epochs_range = range(len(train_losses))
    beta_values = [beta_schedule(e) for e in epochs_range]
    plt.plot(epochs_range, beta_values, label='Beta Schedule', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Beta Value')
    plt.title('Beta Annealing Schedule')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test final model
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_flexible_vae_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on test set
    model.eval()
    test_loss = 0
    test_samples = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            inputs, targets, cif_ids = batch_data
            
            if len(inputs) == 5:  # Has extra features
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea = inputs
            else:  # Structure only
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                extra_fea = None
            
            if args.cuda:
                atom_fea = atom_fea.cuda()
                nbr_fea = nbr_fea.cuda()
                nbr_fea_idx = nbr_fea_idx.cuda()
                crystal_atom_idx = [idx.cuda() for idx in crystal_atom_idx]
                if extra_fea is not None:
                    extra_fea = extra_fea.cuda()
            
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            
            # Simple test loss calculation
            if model.use_extra_features and extra_fea is not None:
                recon_loss = torch.nn.functional.mse_loss(
                    output['reconstructed']['extra_features'], extra_fea
                )
            else:
                recon_loss = torch.tensor(0.0)
            
            kl_loss = -0.5 * torch.sum(
                1 + output['logvar'] - output['mu'].pow(2) - output['logvar'].exp()
            )
            
            test_loss += (recon_loss + kl_loss).item()
            test_samples += 1
    
    test_loss /= test_samples
    print(f"Test Loss: {test_loss:.4f}")
    
    # Generate samples if requested
    if args.generate_samples > 0:
        print(f"\nGenerating {args.generate_samples} new structures...")
        
        generated_samples = model.generate(
            num_samples=args.generate_samples,
            device=device
        )
        
        print("Generated samples summary:")
        print(f"  Number of samples: {args.generate_samples}")
        
        if model.use_extra_features:
            print("  Extra features shape:", generated_samples['extra_features'].shape)
            print("  Sample extra features (first 3):")
            for i in range(min(3, args.generate_samples)):
                features = generated_samples['extra_features'][i].cpu().numpy()
                if hasattr(dataset, 'feature_names') and len(dataset.feature_names) > 0:
                    feature_str = ", ".join([f"{name}: {val:.4f}" 
                                           for name, val in zip(dataset.feature_names[:5], features[:5])])
                    print(f"    Sample {i+1}: {feature_str}...")
                else:
                    print(f"    Sample {i+1}: {features[:5]}...")
        
        print("  Atom count range:", 
              f"{generated_samples['num_atoms'].min().item()} - {generated_samples['num_atoms'].max().item()}")
        
        # Save generated samples
        torch.save(generated_samples, os.path.join(args.save_dir, 'generated_samples.pth'))
        print(f"Generated samples saved to {args.save_dir}/generated_samples.pth")
    
    # Save training summary
    summary = {
        'dataset_info': dataset_info,
        'model_config': {
            'latent_dim': args.latent_dim,
            'atom_fea_len': args.atom_fea_len,
            'h_fea_len': args.h_fea_len,
            'n_conv': args.n_conv,
            'use_extra_features': model.use_extra_features,
            'n_extra_features': model.n_extra_features
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'beta_max': args.beta_max,
            'beta_annealing_epochs': args.beta_annealing_epochs
        },
        'results': {
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'training_time_minutes': total_training_time / 60,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    }
    
    import json
    with open(os.path.join(args.save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to {args.save_dir}/training_summary.json")
    print(f"Model checkpoints saved to {args.save_dir}/")
    
    print("\n" + "="*60)
    print("FLEXIBLE VAE TRAINING COMPLETE!")
    print("="*60)
    
    return model, dataset


if __name__ == '__main__':
    main()
