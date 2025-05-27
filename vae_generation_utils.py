#!/usr/bin/env python3
"""
Generation and Analysis Tools for Flexible Crystal Graph VAE
Handles conditional generation, inverse design, and structure analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple


class CrystalVAEGenerator:
    """
    Generator class for creating new crystal structures using trained VAE
    Handles both structure-only and feature-enabled models
    """
    
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.feature_stats = None
        self.load_model()
    
    def load_model(self):
        """Load trained VAE model and extract metadata"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model parameters from checkpoint
        self.feature_names = checkpoint.get('feature_names', [])
        self.use_extra_features = checkpoint.get('use_extra_features', False)
        self.n_extra_features = checkpoint.get('n_extra_features', 0)
        latent_dim = checkpoint.get('latent_dim', 64)
        dataset_info = checkpoint.get('dataset_info', {})
        
        # Create model with correct architecture
        from flexible_vae_model import FlexibleCrystalGraphVAE
        
        # Use dataset info to set correct dimensions
        orig_atom_fea_len = dataset_info.get('atom_feature_dim', 92)
        nbr_fea_len = dataset_info.get('bond_feature_dim', 41)
        
        self.model = FlexibleCrystalGraphVAE(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=128,
            n_conv=3,
            h_fea_len=256,
            n_extra_features=self.n_extra_features,
            latent_dim=latent_dim,
            max_atoms=200,
            use_extra_features=self.use_extra_features
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded VAE model:")
        print(f"  Model type: {'With extra features' if self.use_extra_features else 'Structure-only'}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Number of features: {self.n_extra_features}")
        if self.feature_names:
            print(f"  Feature names: {', '.join(self.feature_names[:5])}{'...' if len(self.feature_names) > 5 else ''}")
    
    def generate_random_samples(self, num_samples: int = 10, seed: Optional[int] = None) -> List[Dict]:
        """Generate random crystal structures"""
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            generated = self.model.generate(num_samples=num_samples, device=self.device)
        
        return self.process_generated_samples(generated)
    
    def generate_conditional_samples(self, target_features: List[float], 
                                   num_samples: int = 10, max_iterations: int = 1000,
                                   tolerance: float = 0.1) -> List[Dict]:
        """
        Generate samples conditioned on target extra features
        Uses iterative optimization in latent space
        """
        if not self.use_extra_features:
            raise ValueError("Conditional generation requires a model trained with extra features")
        
        target_features = torch.tensor(target_features, dtype=torch.float32, device=self.device)
        if target_features.dim() == 1:
            target_features = target_features.unsqueeze(0)
        
        best_samples = []
        
        print(f"Generating {num_samples} samples targeting specific features...")
        
        for sample_idx in range(num_samples):
            best_z = None
            best_error = float('inf')
            
            # Try multiple random starting points
            for attempt in range(50):
                z = torch.randn(1, self.model.latent_dim, device=self.device) * 0.5
                
                with torch.no_grad():
                    generated = self.model.decode(z)
                    if 'extra_features' in generated:
                        predicted_features = generated['extra_features']
                        
                        # Calculate error
                        error = torch.mean((predicted_features - target_features) ** 2).item()
                        
                        if error < best_error:
                            best_error = error
                            best_z = z.clone()
            
            # Refine using gradient descent if needed
            if best_error > tolerance and best_z is not None:
                best_z = self.refine_latent_vector(best_z, target_features, max_iterations=200)
                
                # Recalculate error
                with torch.no_grad():
                    refined_generated = self.model.decode(best_z)
                    if 'extra_features' in refined_generated:
                        best_error = torch.mean(
                            (refined_generated['extra_features'] - target_features) ** 2
                        ).item()
            
            # Generate final sample with best latent vector
            if best_z is not None:
                with torch.no_grad():
                    best_sample = self.model.decode(best_z)
                    
                    sample_dict = {
                        'latent_vector': best_z.cpu(),
                        'generated_features': best_sample['extra_features'].cpu(),
                        'target_features': target_features.cpu(),
                        'error': best_error,
                        'num_atoms': best_sample['num_atoms'].cpu(),
                        'atom_features': best_sample['atom_features'].cpu(),
                        'atom_types': best_sample['atom_types'].cpu()
                    }
                    
                    # Add feature dictionary if feature names are available
                    if self.feature_names:
                        features = best_sample['extra_features'][0].cpu().numpy()
                        sample_dict['feature_dict'] = {
                            name: value for name, value in zip(self.feature_names, features)
                        }
                    
                    best_samples.append(sample_dict)
                    
                    if (sample_idx + 1) % 5 == 0:
                        print(f"  Generated {sample_idx + 1}/{num_samples} samples, current error: {best_error:.6f}")
        
        print(f"Conditional generation complete. Average error: {np.mean([s['error'] for s in best_samples]):.6f}")
        return best_samples
    
    def refine_latent_vector(self, z_init: torch.Tensor, target_features: torch.Tensor, 
                           max_iterations: int = 200, lr: float = 0.01) -> torch.Tensor:
        """
        Refine latent vector using gradient descent to better match target features
        """
        z = z_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Generate features from current latent vector
            generated = self.model.decode(z)
            if 'extra_features' not in generated:
                break
                
            predicted_features = generated['extra_features']
            
            # Calculate loss
            loss = torch.mean((predicted_features - target_features) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < 1e-6:
                break
        
        return z.detach()
    
    def interpolate_between_samples(self, sample1_features: List[float], 
                                  sample2_features: List[float], 
                                  num_steps: int = 10) -> List[Dict]:
        """
        Interpolate between two sets of features in latent space
        """
        if not self.use_extra_features:
            raise ValueError("Interpolation requires a model trained with extra features")
        
        # Find latent representations of both samples
        z1 = self.find_latent_for_features(sample1_features)
        z2 = self.find_latent_for_features(sample2_features)
        
        # Create interpolation path
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        interpolated_samples = []
        
        with torch.no_grad():
            for alpha in alphas:
                z_interp = (1 - alpha) * z1 + alpha * z2
                generated = self.model.decode(z_interp)
                
                sample_dict = {
                    'alpha': alpha.item(),
                    'latent_vector': z_interp.cpu(),
                    'num_atoms': generated['num_atoms'].cpu(),
                    'atom_features': generated['atom_features'].cpu()
                }
                
                if 'extra_features' in generated:
                    sample_dict['generated_features'] = generated['extra_features'].cpu()
                    
                    if self.feature_names:
                        features = generated['extra_features'][0].cpu().numpy()
                        sample_dict['feature_dict'] = {
                            name: value for name, value in zip(self.feature_names, features)
                        }
                
                interpolated_samples.append(sample_dict)
        
        return interpolated_samples
    
    def find_latent_for_features(self, target_features: List[float], 
                               max_iterations: int = 1000, lr: float = 0.01) -> torch.Tensor:
        """
        Find latent vector that produces target features using gradient descent
        """
        target_features = torch.tensor(target_features, dtype=torch.float32, device=self.device)
        if target_features.dim() == 1:
            target_features = target_features.unsqueeze(0)
        
        # Initialize latent vector
        z = torch.randn(1, self.model.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=lr)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Generate features from current latent vector
            generated = self.model.decode(z)
            if 'extra_features' not in generated:
                break
                
            predicted_features = generated['extra_features']
            
            # Calculate loss
            loss = torch.mean((predicted_features - target_features) ** 2)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss.item() < 1e-6:
                break
        
        return z.detach()
    
    def process_generated_samples(self, generated_batch: Dict) -> List[Dict]:
        """Process generated samples into a more usable format"""
        batch_size = generated_batch['num_atoms'].shape[0]
        processed_samples = []
        
        for i in range(batch_size):
            sample = {
                'num_atoms': generated_batch['num_atoms'][i].item(),
                'atom_features': generated_batch['atom_features'][i].cpu().numpy(),
                'atom_types': generated_batch['atom_types'][i].cpu().numpy()
            }
            
            # Add extra features if available
            if 'extra_features' in generated_batch:
                sample['extra_features'] = generated_batch['extra_features'][i].cpu().numpy()
                
                # Add feature names if available
                if self.feature_names:
                    sample['feature_dict'] = {
                        name: value for name, value in 
                        zip(self.feature_names, sample['extra_features'])
                    }
            
            processed_samples.append(sample)
        
        return processed_samples
    
    def analyze_feature_space(self, samples: List[Dict]) -> Dict[str, Any]:
        """Analyze the distribution of generated features"""
        if not samples or not self.use_extra_features:
            return None
        
        # Extract features
        features_matrix = np.array([sample['extra_features'] for sample in samples])
        
        # Calculate statistics
        stats = {
            'mean': np.mean(features_matrix, axis=0),
            'std': np.std(features_matrix, axis=0),
            'min': np.min(features_matrix, axis=0),
            'max': np.max(features_matrix, axis=0),
            'median': np.median(features_matrix, axis=0)
        }
        
        # Create DataFrame for easier analysis
        if self.feature_names:
            df = pd.DataFrame(features_matrix, columns=self.feature_names)
            
            # Feature correlations
            correlations = df.corr()
            
            return {
                'statistics': stats,
                'dataframe': df,
                'correlations': correlations,
                'feature_names': self.feature_names
            }
        
        return {'statistics': stats, 'features_matrix': features_matrix}


def visualize_generated_samples(samples: List[Dict], feature_names: List[str] = None, 
                               save_path: str = None):
    """
    Create comprehensive visualizations of generated samples
    """
    if not samples:
        print("No samples to visualize")
        return
    
    # Check if samples have extra features
    has_extra_features = 'extra_features' in samples[0]
    
    if has_extra_features:
        # Extract features
        features_matrix = np.array([sample['extra_features'] for sample in samples])
        num_features = features_matrix.shape[1]
        
        # Set up the plot
        fig_height = max(12, (num_features // 3 + 1) * 4)
        fig, axes = plt.subplots((num_features + 2) // 3, 3, figsize=(15, fig_height))
        if num_features == 1:
            axes = [axes]
        elif axes.ndim == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Plot each feature distribution
        for i in range(num_features):
            ax = axes[i] if num_features > 1 else axes[0]
            feature_values = features_matrix[:, i]
            
            # Histogram
            ax.hist(feature_values, bins=20, alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
            
            # Labels
            title = feature_names[i] if feature_names and i < len(feature_names) else f'Feature {i+1}'
            ax.set_title(title)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Feature correlation heatmap
        if num_features > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = np.corrcoef(features_matrix.T)
            
            # Create labels
            labels = feature_names if feature_names else [f'Feature {i+1}' for i in range(num_features)]
            
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       xticklabels=labels, yticklabels=labels,
                       square=True, linewidths=0.5)
            
            plt.title('Generated Features Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                corr_save_path = save_path.replace('.png', '_correlations.png')
                plt.savefig(corr_save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
    
    # Visualize atom count distribution
    atom_counts = [sample['num_atoms'] for sample in samples]
    
    plt.figure(figsize=(10, 6))
    plt.hist(atom_counts, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Generated Structure Size Distribution')
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    mean_atoms = np.mean(atom_counts)
    plt.axvline(mean_atoms, color='red', linestyle='--', label=f'Mean: {mean_atoms:.1f}')
    plt.legend()
    
    if save_path:
        atoms_save_path = save_path.replace('.png', '_atom_counts.png')
        plt.savefig(atoms_save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_latent_space(model_path: str, dataloader, device='cpu', save_dir: str = './analysis',
                        max_samples: int = 1000):
    """
    Analyze the latent space of a trained VAE model
    """
    # Load model
    generator = CrystalVAEGenerator(model_path, device)
    model = generator.model
    
    model.eval()
    latent_vectors = []
    extra_features = [] if generator.use_extra_features else None
    crystal_ids = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            inputs, targets, cif_ids = batch_data
            
            if len(inputs) == 5:  # Has extra features
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea = inputs
            else:  # Structure only
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                extra_fea = None
            
            # Move to device
            if device != 'cpu':
                atom_fea = atom_fea.to(device)
                nbr_fea = nbr_fea.to(device)
                nbr_fea_idx = nbr_fea_idx.to(device)
                crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
                if extra_fea is not None:
                    extra_fea = extra_fea.to(device)
            
            # Encode to latent space
            mu, logvar = model.encode(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            
            latent_vectors.append(mu.cpu())
            if generator.use_extra_features and extra_fea is not None:
                extra_features.append(extra_fea.cpu())
            crystal_ids.extend(cif_ids)
            
            sample_count += len(cif_ids)
    
    # Concatenate vectors
    latent_vectors = torch.cat(latent_vectors, dim=0).numpy()
    if generator.use_extra_features and extra_features:
        extra_features = torch.cat(extra_features, dim=0).numpy()
    
    print(f"Analyzing {latent_vectors.shape[0]} samples in latent space...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Dimensionality reduction
    pca = PCA(n_components=min(10, latent_vectors.shape[1]))
    latent_pca = pca.fit_transform(latent_vectors)
    
    # t-SNE (if reasonable number of samples)
    if latent_vectors.shape[0] <= 2000:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, latent_vectors.shape[0]-1))
        latent_tsne = tsne.fit_transform(latent_vectors)
    else:
        latent_tsne = None
        print("Skipping t-SNE for large dataset (>2000 samples)")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # PCA plot
    ax1 = plt.subplot(2, 4, 1)
    plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.6, s=20)
    plt.title('Latent Space PCA')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(True, alpha=0.3)
    
    # t-SNE plot
    if latent_tsne is not None:
        ax2 = plt.subplot(2, 4, 2)
        plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], alpha=0.6, s=20)
        plt.title('Latent Space t-SNE')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, alpha=0.3)
    
    # Latent distribution
    ax3 = plt.subplot(2, 4, 3)
    plt.hist(latent_vectors.flatten(), bins=50, alpha=0.7, density=True)
    plt.title('Latent Space Distribution')
    plt.xlabel('Latent Values')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # Latent dimension variances
    ax4 = plt.subplot(2, 4, 4)
    latent_std = np.std(latent_vectors, axis=0)
    plt.bar(range(len(latent_std)), latent_std)
    plt.title('Latent Dimension Std Dev')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    # PCA explained variance
    ax5 = plt.subplot(2, 4, 5)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'bo-')
    plt.title('PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    
    # Feature-latent correlations (if extra features available)
    if generator.use_extra_features and extra_features is not None:
        correlations = np.corrcoef(latent_vectors.T, extra_features.T)
        latent_dim = latent_vectors.shape[1]
        feature_correlations = correlations[:latent_dim, latent_dim:]
        
        ax6 = plt.subplot(2, 4, 6)
        im = plt.imshow(feature_correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.title('Latent-Feature Correlations')
        plt.xlabel('Extra Features')
        plt.ylabel('Latent Dimensions')
        if generator.feature_names and len(generator.feature_names) <= 20:
            plt.xticks(range(len(generator.feature_names)), generator.feature_names, rotation=45, ha='right')
        plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_space_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save analysis data
    analysis_data = {
        'latent_vectors': latent_vectors,
        'latent_pca': latent_pca,
        'pca_explained_variance': pca.explained_variance_ratio_,
        'crystal_ids': crystal_ids,
        'use_extra_features': generator.use_extra_features,
        'feature_names': generator.feature_names
    }
    
    if latent_tsne is not None:
        analysis_data['latent_tsne'] = latent_tsne
    
    if generator.use_extra_features and extra_features is not None:
        analysis_data['extra_features'] = extra_features
        analysis_data['feature_correlations'] = feature_correlations
    
    np.savez(os.path.join(save_dir, 'latent_analysis.npz'), **analysis_data)
    print(f"Latent analysis saved to {save_dir}/latent_analysis.npz")
    
    return analysis_data


def save_generated_structures(samples: List[Dict], output_dir: str = './generated_structures',
                            feature_names: List[str] = None):
    """
    Save generated crystal structures in various formats
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle
    with open(os.path.join(output_dir, 'generated_samples.pkl'), 'wb') as f:
        pickle.dump(samples, f)
    
    # Save basic info as text
    with open(os.path.join(output_dir, 'generation_summary.txt'), 'w') as f:
        f.write(f"Generated {len(samples)} crystal structures\n")
        f.write(f"Timestamp: {pd.Timestamp.now()}\n\n")
        
        # Atom count statistics
        atom_counts = [sample['num_atoms'] for sample in samples]
        f.write(f"Atom count statistics:\n")
        f.write(f"  Mean: {np.mean(atom_counts):.1f}\n")
        f.write(f"  Std: {np.std(atom_counts):.1f}\n")
        f.write(f"  Range: {min(atom_counts)} - {max(atom_counts)}\n\n")
        
        # Feature statistics (if available)
        if samples and 'extra_features' in samples[0]:
            features_matrix = np.array([sample['extra_features'] for sample in samples])
            f.write(f"Feature statistics:\n")
            
            if feature_names:
                for i, name in enumerate(feature_names):
                    values = features_matrix[:, i]
                    f.write(f"  {name}:\n")
                    f.write(f"    Mean: {np.mean(values):.4f}\n")
                    f.write(f"    Std: {np.std(values):.4f}\n")
                    f.write(f"    Range: {np.min(values):.4f} - {np.max(values):.4f}\n")
            else:
                f.write(f"  Number of features: {features_matrix.shape[1]}\n")
                f.write(f"  Feature means: {np.mean(features_matrix, axis=0)}\n")
                f.write(f"  Feature stds: {np.std(features_matrix, axis=0)}\n")
    
    # Save features as CSV (if available)
    if samples and 'extra_features' in samples[0]:
        features_matrix = np.array([sample['extra_features'] for sample in samples])
        
        if feature_names:
            df = pd.DataFrame(features_matrix, columns=feature_names)
        else:
            df = pd.DataFrame(features_matrix, columns=[f'Feature_{i}' for i in range(features_matrix.shape[1])])
        
        # Add sample IDs and atom counts
        df.insert(0, 'Sample_ID', [f'Generated_{i:04d}' for i in range(len(samples))])
        df['Num_Atoms'] = [sample['num_atoms'] for sample in samples]
        
        df.to_csv(os.path.join(output_dir, 'generated_features.csv'), index=False)
    
    print(f"Generated structures saved to {output_dir}/")
    print(f"Files created:")
    print(f"  - generated_samples.pkl (full data)")
    print(f"  - generation_summary.txt (summary)")
    if samples and 'extra_features' in samples[0]:
        print(f"  - generated_features.csv (features table)")


# Example usage functions
def example_conditional_generation():
    """
    Example of conditional generation targeting specific properties
    """
    # Load trained model
    model_path = './flexible_vae_checkpoints/best_flexible_vae_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the VAE first using flexible_vae_training.py")
        return
    
    generator = CrystalVAEGenerator(model_path)
    
    if not generator.use_extra_features:
        print("This model was trained without extra features.")
        print("Conditional generation requires extra features.")
        return
    
    # Example: Target high aromatic overlap and close stacking
    print("Example: Generating coronene dimers with high aromatic overlap")
    
    # Define target features (adjust based on your feature names)
    target_features = [
        4.0,    # Centroid_Distance: Close stacking
        5.0,    # Plane_Angle: Nearly parallel
        1.0,    # Parallel_Offset: Minimal offset
        2.0,    # Vertical_Distance: Close contact
        2.0,    # Group1_Planarity: Good planarity
        2.0,    # Group2_Planarity: Good planarity
        0.12,   # Aromatic_Overlap: High overlap
        0.0,    # Stacking_Type_Code: Face-to-face
        4.0,    # Group1_Ring_Path_Avg
        2.0,    # Group2_Ring_Path_Avg
        0.10,   # Sum_5to6_Ratio
        1.5     # Size_Ratio_G1/G2
    ]
    
    # Generate structures
    optimized_structures = generator.generate_conditional_samples(
        target_features=target_features,
        num_samples=10
    )
    
    # Analyze results
    print("\nGenerated structures with target properties:")
    for i, structure in enumerate(optimized_structures):
        print(f"\nStructure {i+1}:")
        print(f"  Error: {structure['error']:.6f}")
        print(f"  Atoms: {structure['num_atoms'].item()}")
        
        if 'feature_dict' in structure:
            print("  Key features:")
            feature_dict = structure['feature_dict']
            print(f"    Centroid_Distance: {feature_dict.get('Centroid_Distance', 'N/A'):.3f}")
            print(f"    Aromatic_Overlap: {feature_dict.get('Aromatic_Overlap', 'N/A'):.4f}")
            print(f"    Plane_Angle: {feature_dict.get('Plane_Angle', 'N/A'):.1f}")
    
    # Save results
    save_generated_structures(optimized_structures, './conditional_generation_results', generator.feature_names)
    
    return optimized_structures


if __name__ == '__main__':
    # Run example
    example_conditional_generation()
