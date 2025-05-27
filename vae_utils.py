#!/usr/bin/env python3
"""
Utilities and examples for handling any number of features with the Flexible VAE
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import os


def analyze_feature_requirements(feature_file: str) -> Dict[str, Any]:
    """
    Analyze a feature file to determine optimal VAE architecture parameters
    """
    if not os.path.exists(feature_file):
        return {"error": f"Feature file {feature_file} not found"}
    
    # Read the feature file
    try:
        df = pd.read_csv(feature_file)
        feature_columns = df.columns[1:]  # Skip ID column
        num_features = len(feature_columns)
        num_samples = len(df)
        
        print(f"Feature File Analysis: {feature_file}")
        print("=" * 50)
        print(f"Number of features: {num_features}")
        print(f"Number of samples: {num_samples}")
        print(f"Feature names: {list(feature_columns)}")
        
        # Calculate feature statistics
        feature_stats = {}
        for col in feature_columns:
            values = df[col].values
            feature_stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'missing_count': df[col].isna().sum()
            }
        
        # Determine recommended architecture
        recommendations = get_architecture_recommendations(num_features, feature_stats)
        
        analysis_result = {
            'num_features': num_features,
            'num_samples': num_samples,
            'feature_names': list(feature_columns),
            'feature_stats': feature_stats,
            'recommendations': recommendations,
            'complexity_category': categorize_feature_complexity(num_features, feature_stats)
        }
        
        print(f"\nComplexity Category: {analysis_result['complexity_category']}")
        print(f"Recommended Architecture:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        return analysis_result
        
    except Exception as e:
        return {"error": f"Error analyzing feature file: {str(e)}"}


def categorize_feature_complexity(num_features: int, feature_stats: Dict) -> str:
    """
    Categorize the complexity of a feature set
    """
    if num_features <= 3:
        return "minimal"
    elif num_features <= 10:
        return "small"
    elif num_features <= 25:
        return "medium"
    elif num_features <= 50:
        return "large"
    elif num_features <= 100:
        return "very_large"
    else:
        return "massive"


def get_architecture_recommendations(num_features: int, feature_stats: Dict) -> Dict[str, Any]:
    """
    Get recommended VAE architecture parameters based on feature analysis
    """
    complexity = categorize_feature_complexity(num_features, feature_stats)
    
    recommendations = {
        "minimal": {
            "latent_dim": 16,
            "h_fea_len": 128,
            "atom_fea_len": 64,
            "n_conv": 2,
            "batch_size": 64,
            "lr": 0.002,
            "epochs": 50,
            "beta_annealing_epochs": 20
        },
        "small": {
            "latent_dim": 32,
            "h_fea_len": 256,
            "atom_fea_len": 128,
            "n_conv": 3,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 100,
            "beta_annealing_epochs": 40
        },
        "medium": {
            "latent_dim": 64,
            "h_fea_len": 512,
            "atom_fea_len": 128,
            "n_conv": 3,
            "batch_size": 32,
            "lr": 0.001,
            "epochs": 150,
            "beta_annealing_epochs": 60
        },
        "large": {
            "latent_dim": 128,
            "h_fea_len": 512,
            "atom_fea_len": 256,
            "n_conv": 4,
            "batch_size": 16,
            "lr": 0.0005,
            "epochs": 200,
            "beta_annealing_epochs": 80
        },
        "very_large": {
            "latent_dim": 256,
            "h_fea_len": 1024,
            "atom_fea_len": 256,
            "n_conv": 4,
            "batch_size": 8,
            "lr": 0.0005,
            "epochs": 300,
            "beta_annealing_epochs": 100
        },
        "massive": {
            "latent_dim": 512,
            "h_fea_len": 1024,
            "atom_fea_len": 512,
            "n_conv": 5,
            "batch_size": 4,
            "lr": 0.0002,
            "epochs": 500,
            "beta_annealing_epochs": 150
        }
    }
    
    base_rec = recommendations[complexity].copy()
    
    # Adjust based on feature characteristics
    if feature_stats:
        ranges = [stats['range'] for stats in feature_stats.values()]
        max_range = max(ranges) if ranges else 1.0
        
        # If features have very different scales, suggest larger hidden dimensions
        if max_range > 1000:
            base_rec["h_fea_len"] = min(base_rec["h_fea_len"] * 2, 2048)
            base_rec["lr"] *= 0.5  # Use smaller learning rate for complex features
    
    return base_rec


def create_feature_examples():
    """
    Create example feature files with different numbers of features
    """
    examples = {
        "3_features": {
            "names": ["Energy", "Volume", "Density"],
            "description": "Minimal set - basic material properties"
        },
        "8_features": {
            "names": ["Energy", "Volume", "Density", "BandGap", "Elasticity", 
                     "ThermalConductivity", "ElectronicConductivity", "MagneticMoment"],
            "description": "Small set - common material properties"
        },
        "15_features": {
            "names": ["Energy", "Volume", "Density", "BandGap", "Elasticity", 
                     "ThermalConductivity", "ElectronicConductivity", "MagneticMoment",
                     "Hardness", "MeltingPoint", "BoilingPoint", "Enthalpy", 
                     "Entropy", "HeatCapacity", "ThermalExpansion"],
            "description": "Medium set - comprehensive material properties"
        },
        "30_features": {
            "names": [f"Property_{i:02d}" for i in range(1, 31)],
            "description": "Large set - extensive property characterization"
        },
        "100_features": {
            "names": [f"Feature_{i:03d}" for i in range(1, 101)],
            "description": "Very large set - high-dimensional characterization"
        }
    }
    
    print("Feature Set Examples:")
    print("=" * 50)
    
    for example_name, info in examples.items():
        num_features = len(info["names"])
        complexity = categorize_feature_complexity(num_features, {})
        
        print(f"\n{example_name.upper().replace('_', ' ')}:")
        print(f"  Features: {num_features}")
        print(f"  Complexity: {complexity}")
        print(f"  Description: {info['description']}")
        print(f"  Sample features: {', '.join(info['names'][:5])}")
        if num_features > 5:
            print(f"  ... and {num_features - 5} more")
        
        # Show recommended architecture
        recommendations = get_architecture_recommendations(num_features, {})
        print(f"  Recommended latent dim: {recommendations['latent_dim']}")
        print(f"  Recommended hidden dim: {recommendations['h_fea_len']}")
    
    return examples


def generate_synthetic_feature_file(num_features: int, num_samples: int = 1000, 
                                   output_file: str = None, feature_types: str = "mixed") -> str:
    """
    Generate a synthetic feature file for testing with any number of features
    """
    if output_file is None:
        output_file = f"synthetic_{num_features}_features.csv"
    
    print(f"Generating synthetic feature file: {output_file}")
    print(f"  Features: {num_features}")
    print(f"  Samples: {num_samples}")
    print(f"  Feature types: {feature_types}")
    
    # Generate feature names
    if num_features <= 20:
        # Use meaningful names for small sets
        base_names = ["Energy", "Volume", "Density", "BandGap", "Elasticity", 
                     "ThermalCond", "ElecCond", "MagMoment", "Hardness", "MeltPoint",
                     "BoilPoint", "Enthalpy", "Entropy", "HeatCap", "ThermalExp",
                     "Polarization", "Dielectric", "Refraction", "Absorption", "Emission"]
        feature_names = base_names[:num_features]
        if num_features > len(base_names):
            feature_names.extend([f"Property_{i}" for i in range(len(base_names) + 1, num_features + 1)])
    else:
        # Use systematic names for large sets
        feature_names = [f"Feature_{i:04d}" for i in range(1, num_features + 1)]
    
    # Generate synthetic data
    np.random.seed(42)  # For reproducibility
    data = {"CrystalID": [f"Crystal_{i:06d}" for i in range(1, num_samples + 1)]}
    
    for i, feature_name in enumerate(feature_names):
        if feature_types == "mixed":
            # Create diverse feature types
            if i % 4 == 0:
                # Energy-like (can be negative)
                values = np.random.normal(-5, 2, num_samples)
            elif i % 4 == 1:
                # Distance/size-like (positive, log-normal)
                values = np.random.lognormal(1, 0.5, num_samples)
            elif i % 4 == 2:
                # Ratio/percentage-like (0-1)
                values = np.random.beta(2, 2, num_samples)
            else:
                # Angle/categorical-like (bounded)
                values = np.random.uniform(0, 180, num_samples)
        elif feature_types == "normalized":
            # All features normalized around 0
            values = np.random.normal(0, 1, num_samples)
        elif feature_types == "positive":
            # All features positive
            values = np.random.exponential(2, num_samples)
        else:
            # Default: mixed normal distributions
            values = np.random.normal(i, 1, num_samples)
        
        data[feature_name] = values
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    print(f"Synthetic feature file saved: {output_file}")
    return output_file


def visualize_feature_distribution(feature_file: str, max_features_to_plot: int = 20,
                                  save_path: str = None):
    """
    Visualize the distribution of features (handles any number of features)
    """
    df = pd.read_csv(feature_file)
    feature_columns = df.columns[1:]  # Skip ID column
    num_features = len(feature_columns)
    
    print(f"Visualizing feature distributions for {num_features} features...")
    
    # Determine plot layout
    features_to_plot = min(num_features, max_features_to_plot)
    cols = min(4, features_to_plot)
    rows = (features_to_plot + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot feature distributions
    for i in range(features_to_plot):
        ax = axes[i] if features_to_plot > 1 else axes
        feature_name = feature_columns[i]
        values = df[feature_name].values
        
        # Create histogram
        ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'{feature_name}\n(μ={np.mean(values):.3f}, σ={np.std(values):.3f})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(features_to_plot, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Feature Distributions ({num_features} total features, showing first {features_to_plot})', 
                fontsize=14, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution plot saved: {save_path}")
    
    plt.show()
    
    # Create correlation heatmap for reasonable number of features
    if num_features <= 50:
        plt.figure(figsize=(max(8, num_features * 0.4), max(6, num_features * 0.3)))
        
        # Calculate correlation matrix
        corr_matrix = df[feature_columns].corr()
        
        # Create heatmap
        sns.heatmap(corr_matrix, 
                   annot=num_features <= 15,  # Only annotate for small sets
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   xticklabels=True,
                   yticklabels=True)
        
        plt.title(f'Feature Correlation Matrix ({num_features} features)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            corr_save_path = save_path.replace('.png', '_correlations.png')
            plt.savefig(corr_save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved: {corr_save_path}")
        
        plt.show()
    else:
        print(f"Skipping correlation heatmap for {num_features} features (too many to visualize)")


def create_training_commands_for_different_feature_counts():
    """
    Create example training commands for different feature counts
    """
    examples = [
        (3, "minimal dataset"),
        (8, "small dataset"),
        (15, "medium dataset"), 
        (30, "large dataset"),
        (50, "very large dataset"),
        (100, "massive dataset")
    ]
    
    print("TRAINING COMMANDS FOR DIFFERENT FEATURE COUNTS")
    print("=" * 80)
    
    for num_features, description in examples:
        recommendations = get_architecture_recommendations(num_features, {})
        
        print(f"\n{num_features} FEATURES ({description.upper()}):")
        print("-" * 60)
        
        command = f"""python flexible_vae_training.py ./data/MyData/ \\
    --feature-file ./data/features_{num_features}.csv \\
    --latent-dim {recommendations['latent_dim']} \\
    --epochs {recommendations['epochs']} \\
    --batch-size {recommendations['batch_size']} \\
    --lr {recommendations['lr']} \\
    --beta-annealing-epochs {recommendations['beta_annealing_epochs']} \\
    --atom-fea-len {recommendations['atom_fea_len']} \\
    --h-fea-len {recommendations['h_fea_len']} \\
    --n-conv {recommendations['n_conv']} \\
    --save-dir ./vae_{num_features}_features \\
    --generate-samples 20"""
        
        print(command)
        
        complexity = categorize_feature_complexity(num_features, {})
        print(f"\nComplexity: {complexity}")
        print(f"Estimated training time: {estimate_training_time(recommendations)}")
        print(f"Memory requirements: {estimate_memory_requirements(recommendations)}")


def estimate_training_time(recommendations: Dict) -> str:
    """
    Estimate training time based on architecture complexity
    """
    base_time = 2  # minutes per epoch baseline
    
    # Scale by model size
    size_factor = (recommendations['h_fea_len'] / 256) * (recommendations['latent_dim'] / 64)
    conv_factor = recommendations['n_conv'] / 3
    epoch_factor = recommendations['epochs'] / 100
    
    estimated_minutes = base_time * size_factor * conv_factor * epoch_factor
    
    if estimated_minutes < 60:
        return f"~{estimated_minutes:.0f} minutes"
    elif estimated_minutes < 1440:
        return f"~{estimated_minutes/60:.1f} hours"
    else:
        return f"~{estimated_minutes/1440:.1f} days"


def estimate_memory_requirements(recommendations: Dict) -> str:
    """
    Estimate memory requirements based on architecture
    """
    # Rough estimation based on model parameters
    base_memory = 2  # GB baseline
    
    size_factor = (recommendations['h_fea_len'] / 256) * (recommendations['atom_fea_len'] / 128)
    batch_factor = recommendations['batch_size'] / 32
    
    estimated_gb = base_memory * size_factor * batch_factor
    
    if estimated_gb < 4:
        return "Low (~2-4 GB)"
    elif estimated_gb < 8:
        return "Medium (~4-8 GB)"
    elif estimated_gb < 16:
        return "High (~8-16 GB)"
    else:
        return "Very High (16+ GB)"


def demo_flexible_feature_handling():
    """
    Demonstrate the flexible feature handling capabilities
    """
    print("FLEXIBLE FEATURE HANDLING DEMONSTRATION")
    print("=" * 80)
    
    # Create examples with different feature counts
    feature_counts = [1, 3, 8, 15, 25, 50, 100]
    
    print("\n1. Creating synthetic datasets with different feature counts...")
    for count in feature_counts:
        filename = generate_synthetic_feature_file(
            num_features=count, 
            num_samples=100,  # Small for demo
            output_file=f"demo_{count}_features.csv"
        )
        
        # Analyze the created file
        analysis = analyze_feature_requirements(filename)
        print(f"  ✓ {count} features: {analysis['complexity_category']} complexity")
    
    print("\n2. Architecture recommendations:")
    for count in feature_counts:
        recommendations = get_architecture_recommendations(count, {})
        complexity = categorize_feature_complexity(count, {})
        print(f"  {count:3d} features ({complexity:>10}): "
              f"latent_dim={recommendations['latent_dim']:3d}, "
              f"h_fea_len={recommendations['h_fea_len']:4d}")
    
    print("\n3. Training command examples:")
    create_training_commands_for_different_feature_counts()
    
    print("\n4. Feature visualization examples:")
    # Visualize a few examples
    for count in [3, 15]:
        filename = f"demo_{count}_features.csv"
        if os.path.exists(filename):
            print(f"\nVisualizing {count} features...")
            visualize_feature_distribution(
                filename, 
                save_path=f"demo_{count}_features_dist.png"
            )
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("The system handles ANY number of features automatically.")
    print("Just provide your feature file and the VAE will adapt!")


if __name__ == '__main__':
    demo_flexible_feature_handling()
