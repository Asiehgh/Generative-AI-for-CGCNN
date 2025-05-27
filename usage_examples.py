#!/usr/bin/env python3
"""
Usage examples for the Ultra-Flexible Crystal Graph VAE
Shows how to use the system with ANY number of features (1, 5, 12, 50, 100+, etc.)
"""

import os
import torch
import sys


def examples_by_feature_count():
    """
    Examples organized by number of features
    """
    print("="*80)
    print("EXAMPLES BY FEATURE COUNT - ANY NUMBER OF FEATURES SUPPORTED")
    print("="*80)
    
    examples = [
        (1, "Single property", ["Energy"]),
        (3, "Basic properties", ["Energy", "Volume", "Density"]),
        (5, "Extended properties", ["Energy", "Volume", "Density", "BandGap", "Magnetization"]),
        (12, "Your coronene dataset", [
            "Centroid_Distance", "Plane_Angle", "Parallel_Offset", "Vertical_Distance",
            "Group1_Planarity", "Group2_Planarity", "Aromatic_Overlap", "Stacking_Type_Code",
            "Group1_Ring_Path_Avg", "Group2_Ring_Path_Avg", "Sum_5to6_Ratio", "Size_Ratio_G1/G2"
        ]),
        (25, "Comprehensive characterization", ["Property_{:02d}".format(i) for i in range(1, 26)]),
        (50, "High-dimensional features", ["Feature_{:03d}".format(i) for i in range(1, 51)]),
        (100, "Very high-dimensional", ["Feature_{:03d}".format(i) for i in range(1, 101)]),
        (500, "Massive feature space", ["Feature_{:04d}".format(i) for i in range(1, 501)])
    ]
    
    for num_features, description, sample_features in examples:
        print(f"\n{num_features:3d} FEATURES - {description.upper()}")
        print("-" * 70)
        
        # Show sample features
        if num_features <= 12:
            print(f"Features: {', '.join(sample_features)}")
        else:
            print(f"Sample features: {', '.join(sample_features[:8])} ... (+{num_features-8} more)")
        
        # Training command
        print("Training command:")
        command = f"""python flexible_vae_training.py ./data/MyData/ \\
    --feature-file ./data/features_{num_features}.csv \\
    --latent-dim {get_recommended_latent_dim(num_features)} \\
    --h-fea-len {get_recommended_hidden_dim(num_features)} \\
    --epochs {get_recommended_epochs(num_features)} \\
    --batch-size {get_recommended_batch_size(num_features)} \\
    --save-dir ./vae_{num_features}_features"""
        
        print(command)
        
        # What the system does automatically
        print("Automatic adaptations:")
        print(f"  ✓ Architecture scales to handle {num_features} features")
        print(f"  ✓ Feature encoder: {num_features} → {get_recommended_hidden_dim(num_features)//4} → latent")
        print(f"  ✓ Feature decoder: latent → {get_recommended_hidden_dim(num_features)//4} → {num_features}")
        print(f"  ✓ Normalization computed for all {num_features} features")
        
        if num_features <= 10:
            print("  ✓ Lightweight architecture for small feature set")
        elif num_features <= 50:
            print("  ✓ Moderate architecture for medium feature set")
        else:
            print("  ✓ Full-capacity architecture for large feature set")


def get_recommended_latent_dim(num_features):
    """Get recommended latent dimension based on feature count"""
    if num_features <= 5:
        return 32
    elif num_features <= 15:
        return 64
    elif num_features <= 30:
        return 128
    elif num_features <= 100:
        return 256
    else:
        return 512


def get_recommended_hidden_dim(num_features):
    """Get recommended hidden dimension based on feature count"""
    if num_features <= 5:
        return 128
    elif num_features <= 15:
        return 256
    elif num_features <= 30:
        return 512
    elif num_features <= 100:
        return 1024
    else:
        return 2048


def get_recommended_epochs(num_features):
    """Get recommended epochs based on feature count"""
    if num_features <= 5:
        return 50
    elif num_features <= 15:
        return 100
    elif num_features <= 30:
        return 150
    elif num_features <= 100:
        return 200
    else:
        return 300


def get_recommended_batch_size(num_features):
    """Get recommended batch size based on feature count"""
    if num_features <= 10:
        return 64
    elif num_features <= 30:
        return 32
    elif num_features <= 100:
        return 16
    else:
        return 8


def real_world_scenarios():
    """
    Real-world scenarios with different feature counts
    """
    print("\n" + "="*80)
    print("REAL-WORLD SCENARIOS - DIFFERENT RESEARCH APPLICATIONS")
    print("="*80)
    
    scenarios = [
        {
            "name": "Drug Discovery - Small Molecules",
            "features": 8,
            "examples": ["LogP", "MW", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "Complexity"],
            "use_case": "Optimize molecular properties for drug-likeness"
        },
        {
            "name": "Your Coronene Dimers",
            "features": 12,
            "examples": ["Centroid_Distance", "Plane_Angle", "Parallel_Offset", "Vertical_Distance", "...8 more"],
            "use_case": "Design new π-π stacking configurations"
        },
        {
            "name": "Perovskite Solar Cells",
            "features": 18,
            "examples": ["BandGap", "FormationEnergy", "EffectiveMass_e", "EffectiveMass_h", "...14 more"],
            "use_case": "Discover new perovskite compositions for photovoltaics"
        },
        {
            "name": "Metal-Organic Frameworks (MOFs)",
            "features": 35,
            "examples": ["SurfaceArea", "PoreVolume", "CO2_Uptake", "H2_Uptake", "...31 more"],
            "use_case": "Design MOFs for gas storage and separation"
        },
        {
            "name": "High-Entropy Alloys",
            "features": 67,
            "examples": ["YoungsModulus", "UltimateStrength", "Hardness", "Density", "...63 more"],
            "use_case": "Optimize mechanical properties for aerospace applications"
        },
        {
            "name": "Comprehensive Materials Database",
            "features": 150,
            "examples": ["Electronic", "Mechanical", "Thermal", "Optical", "Magnetic", "...145 more"],
            "use_case": "Universal materials property prediction and design"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name'].upper()}")
        print("-" * 60)
        print(f"Features: {scenario['features']}")
        print(f"Examples: {', '.join(scenario['examples'])}")
        print(f"Use case: {scenario['use_case']}")
        
        # Automatic system adaptation
        latent_dim = get_recommended_latent_dim(scenario['features'])
        hidden_dim = get_recommended_hidden_dim(scenario['features'])
        
        print(f"Automatic adaptation:")
        print(f"  → Latent dimension: {latent_dim}")
        print(f"  → Hidden dimension: {hidden_dim}")
        print(f"  → Feature processing layers: {get_num_layers(scenario['features'])}")
        
        print(f"Training command:")
        print(f"  python flexible_vae_training.py ./data/{scenario['name'].replace(' ', '_').replace('-', '_')}/ \\")
        print(f"    --feature-file ./features.csv \\")
        print(f"    --latent-dim {latent_dim} --h-fea-len {hidden_dim}")


def get_num_layers(num_features):
    """Get number of processing layers based on feature count"""
    if num_features <= 10:
        return "2 layers (simple)"
    elif num_features <= 30:
        return "3 layers (moderate)"
    elif num_features <= 100:
        return "4 layers (deep)"
    else:
        return "5+ layers (very deep)"


def feature_file_format_examples():
    """
    Show how to format feature files for different numbers of features
    """
    print("\n" + "="*80)
    print("FEATURE FILE FORMAT EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "name": "3 Features (Minimal)",
            "content": """CrystalID,Energy,Volume,Density
Crystal_001,-2.45,150.23,2.67
Crystal_002,-1.89,142.67,2.89
Crystal_003,-3.12,158.45,2.45"""
        },
        {
            "name": "12 Features (Your Coronene)",
            "content": """CrystalID,Centroid_Distance,Plane_Angle,Parallel_Offset,Vertical_Distance,Group1_Planarity,Group2_Planarity,Aromatic_Overlap,Stacking_Type_Code,Group1_Ring_Path_Avg,Group2_Ring_Path_Avg,Sum_5to6_Ratio,Size_Ratio_G1/G2
Coronene_T1000_P10_dimer_100_102,3.736,8.924,1.406,3.461,3.463,1.171,0.069,0.000,3.761,0.519,0.100,1.758738
Coronene_T1000_P10_dimer_100_154,11.523,31.009,11.505,0.627,3.374,5.882,0.011,1.000,3.412,3.783,0.200,0.934626"""
        },
        {
            "name": "50 Features (Large Set)",
            "content": """CrystalID,Feature_001,Feature_002,Feature_003,...,Feature_050
Crystal_001,1.23,4.56,7.89,...,0.12
Crystal_002,2.34,5.67,8.90,...,0.23
Crystal_003,3.45,6.78,9.01,...,0.34"""
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print("-" * 40)
        print(example['content'])
        print()
    
    print("Key Points:")
    print("✓ First column: Crystal ID (any format)")
    print("✓ Remaining columns: Feature values (numeric)")
    print("✓ Header row: Required with feature names")
    print("✓ Any number of features: 1, 5, 12, 50, 100+")
    print("✓ System automatically detects and adapts")


def automatic_adaptation_details():
    """
    Explain how the system automatically adapts to different feature counts
    """
    print("\n" + "="*80)
    print("AUTOMATIC ADAPTATION DETAILS")
    print("="*80)
    
    print("\n1. FEATURE DETECTION:")
    print("   → Reads CSV header to count features")
    print("   → Validates feature format and completeness")
    print("   → Reports feature statistics and ranges")
    
    print("\n2. ARCHITECTURE SCALING:")
    print("   → Encoder: Scales hidden layers based on feature count")
    print("   → Decoder: Mirrors encoder architecture")
    print("   → Latent space: Adapts dimension to feature complexity")
    
    print("\n3. TRAINING ADAPTATION:")
    print("   → Batch size: Smaller for large feature sets")
    print("   → Learning rate: Adjusted for feature complexity")
    print("   → Epochs: More training for complex feature sets")
    
    print("\n4. AUTOMATIC SCALING RULES:")
    
    scaling_rules = [
        ("1-5 features", "Lightweight", "32", "128", "64", "0.002"),
        ("6-15 features", "Small", "64", "256", "32", "0.001"),
        ("16-30 features", "Medium", "128", "512", "32", "0.001"),
        ("31-100 features", "Large", "256", "1024", "16", "0.0005"),
        ("100+ features", "Massive", "512", "2048", "8", "0.0002")
    ]
    
    print(f"{'Feature Range':15} | {'Architecture':12} | {'Latent':6} | {'Hidden':6} | {'Batch':5} | {'LR':6}")
    print("-" * 70)
    for rule in scaling_rules:
        print(f"{rule[0]:15} | {rule[1]:12} | {rule[2]:6} | {rule[3]:6} | {rule[4]:5} | {rule[5]:6}")
    
    print("\n5. WHAT YOU DON'T NEED TO WORRY ABOUT:")
    print("   ✗ Architecture design")
    print("   ✗ Hyperparameter tuning")
    print("   ✗ Feature normalization")
    print("   ✗ Network sizing")
    print("   ✗ Memory management")
    
    print("\n6. WHAT THE SYSTEM HANDLES AUTOMATICALLY:")
    print("   ✓ Optimal architecture for your feature count")
    print("   ✓ Proper feature normalization")
    print("   ✓ Appropriate training parameters")
    print("   ✓ Memory-efficient processing")
    print("   ✓ Quality validation and error checking")


def main():
    """
    Comprehensive demonstration of feature flexibility
    """
    print("ULTRA-FLEXIBLE CRYSTAL GRAPH VAE")
    print("SUPPORTS ANY NUMBER OF FEATURES: 1, 5, 12, 50, 100, 500+")
    print("=" * 80)
    
    examples_by_feature_count()
    real_world_scenarios()
    feature_file_format_examples()
    automatic_adaptation_details()
    
    print("\n" + "="*80)
    print("GETTING STARTED WITH YOUR FEATURE COUNT")
    print("="*80)
    
    print("\nStep 1: Prepare your feature file")
    print("  - CSV format with CrystalID + any number of feature columns")
    print("  - Can have 1, 5, 12, 50, 100, or even 500+ features")
    
    print("\nStep 2: Run the flexible training")
    print("  python flexible_vae_training.py ./data/MyData/ --feature-file ./features.csv")
    
    print("\nStep 3: System automatically:")
    print("  ✓ Detects your feature count")
    print("  ✓ Designs optimal architecture")
    print("  ✓ Sets appropriate hyperparameters")
    print("  ✓ Trains and generates new structures")
    
    print("\nNo matter if you have:")
    print("  • 1 feature (energy only)")
    print("  • 12 features (your coronene case)")
    print("  • 50 features (comprehensive characterization)")
    print("  • 100+ features (high-dimensional research)")
    
    print("\nThe system adapts automatically - just provide your data!")


if __name__ == '__main__':
    main()
    """
    Example: Training VAE with your coronene dimer dataset (with 12 extra features)
    """
    print("="*60)
    print("EXAMPLE 1: VAE WITH EXTRA FEATURES")
    print("="*60)
    
    print("Use case: Your coronene dimer dataset with 12 structural features")
    print("\nTraining command:")
    
    command = """
python flexible_vae_training.py ./data/MyData/ \\
    --feature-file ./data/MyData/features.csv \\
    --latent-dim 64 \\
    --epochs 100 \\
    --batch-size 32 \\
    --lr 0.001 \\
    --beta-max 1.0 \\
    --beta-annealing-epochs 50 \\
    --atom-fea-len 128 \\
    --h-fea-len 256 \\
    --n-conv 3 \\
    --save-dir ./vae_with_features \\
    --generate-samples 20
"""
    
    print(command)
    
    print("\nWhat this does:")
    print("- Automatically detects that you have extra features")
    print("- Creates a VAE that encodes both crystal structure AND your 12 features")
    print("- Learns to generate new structures with corresponding feature values")
    print("- Can do conditional generation (target specific feature combinations)")
    
    print("\nGenerated outputs:")
    print("- New crystal structures")
    print("- Corresponding values for all 12 features:")
    features = [
        "Centroid_Distance", "Plane_Angle", "Parallel_Offset", "Vertical_Distance",
        "Group1_Planarity", "Group2_Planarity", "Aromatic_Overlap", "Stacking_Type_Code",
        "Group1_Ring_Path_Avg", "Group2_Ring_Path_Avg", "Sum_5to6_Ratio", "Size_Ratio_G1/G2"
    ]
    for i, feature in enumerate(features, 1):
        print(f"  {i:2d}. {feature}")


def example_structure_only():
    """
    Example: Training VAE with only crystal structures (no extra features)
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: VAE WITH STRUCTURE ONLY")
    print("="*60)
    
    print("Use case: Dataset with only CIF files, no additional features")
    print("\nTraining command:")
    
    command = """
python flexible_vae_training.py ./data/StructureOnlyData/ \\
    --force-structure-only \\
    --latent-dim 64 \\
    --epochs 100 \\
    --batch-size 32 \\
    --lr 0.001 \\
    --atom-fea-len 128 \\
    --h-fea-len 256 \\
    --n-conv 3 \\
    --save-dir ./vae_structure_only \\
    --generate-samples 20
"""
    
    print(command)
    
    print("\nWhat this does:")
    print("- Automatically adapts to structure-only mode")
    print("- Creates a VAE that only encodes crystal structure information")
    print("- Learns structural patterns and can generate new crystal structures")
    print("- Focuses on atomic arrangements, bond patterns, etc.")
    
    print("\nGenerated outputs:")
    print("- New crystal structures with realistic:")
    print("  • Atomic positions")
    print("  • Bond networks") 
    print("  • Crystal symmetries")
    print("  • Chemical compositions")


def example_automatic_detection():
    """
    Example: Automatic detection of dataset type
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: AUTOMATIC DATASET DETECTION")
    print("="*60)
    
    print("The system automatically detects your dataset type:")
    
    print("\nScenario A: Feature file exists")
    print("Command: python flexible_vae_training.py ./data/MyData/ --feature-file ./features.csv")
    print("→ System detects features and uses full VAE with extra features")
    
    print("\nScenario B: No feature file provided")
    print("Command: python flexible_vae_training.py ./data/MyData/")
    print("→ System automatically switches to structure-only mode")
    
    print("\nScenario C: Feature file provided but doesn't exist")
    print("Command: python flexible_vae_training.py ./data/MyData/ --feature-file ./missing.csv")
    print("→ System warns about missing file and switches to structure-only mode")
    
    print("\nScenario D: Force structure-only mode")
    print("Command: python flexible_vae_training.py ./data/MyData/ --force-structure-only")
    print("→ System ignores any feature files and uses structure-only mode")


def example_generation_and_analysis():
    """
    Example: Using trained models for generation and analysis
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: GENERATION AND ANALYSIS")
    print("="*60)
    
    print("After training, you can use the model for various tasks:")
    
    print("\n1. Random Generation:")
    code = '''
from flexible_cgcnn_vae import FlexibleCrystalGraphVAE
import torch

# Load trained model
model_path = './vae_with_features/best_flexible_vae_model.pth'
checkpoint = torch.load(model_path)
model = FlexibleCrystalGraphVAE(...)  # with appropriate parameters
model.load_state_dict(checkpoint['model_state_dict'])

# Generate new structures
new_structures = model.generate(num_samples=10)

if model.use_extra_features:
    print("Generated features:", new_structures['extra_features'])
print("Generated atom counts:", new_structures['num_atoms'])
'''
    print(code)
    
    print("\n2. Conditional Generation (for models with extra features):")
    code = '''
# Target specific feature values
target_features = [
    5.0,   # Centroid_Distance
    30.0,  # Plane_Angle  
    3.0,   # Parallel_Offset
    2.0,   # Vertical_Distance
    # ... specify all 12 features
]

# This would require implementing conditional generation
# (find latent vectors that produce desired features)
'''
    print(code)
    
    print("\n3. Interpolation between structures:")
    code = '''
# Interpolate between two existing structures
structure1 = dataset[0]  # Get first structure
structure2 = dataset[100]  # Get another structure

interpolated = model.interpolate(structure1, structure2, num_steps=10)
# Creates smooth transition between the two structures
'''
    print(code)


def example_integration_with_existing_code():
    """
    Example: How this integrates with existing CGCNN code
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: INTEGRATION WITH YOUR EXISTING CODE")
    print("="*60)
    
    print("The flexible VAE seamlessly integrates with your existing CGCNN:")
    
    print("\n1. Uses your existing data loading:")
    print("   ✓ Same CIF file format")
    print("   ✓ Same id_prop.csv structure")
    print("   ✓ Same atom_init.json")
    print("   ✓ Compatible with your feature CSV format")
    
    print("\n2. Builds on your model architecture:")
    print("   ✓ Uses your ConvLayer implementation")
    print("   ✓ Same graph convolution approach")
    print("   ✓ Compatible feature processing")
    
    print("\n3. Flexible feature handling:")
    print("   ✓ Works with your 12 coronene features")
    print("   ✓ Works with any number of features")
    print("   ✓ Works with no features at all")
    print("   ✓ Automatic normalization")
    
    print("\n4. Easy switching between modes:")
    print("   • Research with features: Use full VAE")
    print("   • Benchmark datasets: Use structure-only VAE")
    print("   • Quick prototyping: Auto-detection")


def comparison_table():
    """
    Show comparison between different modes
    """
    print("\n" + "="*80)
    print("COMPARISON: WITH FEATURES VS STRUCTURE-ONLY")
    print("="*80)
    
    comparison = [
        ("Aspect", "With Extra Features", "Structure-Only"),
        ("-" * 20, "-" * 25, "-" * 20),
        ("Input Data", "CIF + feature CSV", "CIF files only"),
        ("Latent Space", "Structure + features", "Structure only"),
        ("Generation", "Structure + feature values", "Structure only"),
        ("Use Cases", "Property prediction", "Structure discovery"),
        ("", "Conditional design", "Pattern learning"),
        ("", "Feature optimization", "Crystal synthesis"),
        ("Model Size", "Larger (extra networks)", "Smaller (structure only)"),
        ("Training Time", "Longer (more complex)", "Faster (simpler)"),
        ("Generation Quality", "Guided by properties", "Purely structural"),
        ("Applications", "Drug design, catalysis", "Material discovery"),
        ("", "Property targeting", "Structural diversity")
    ]
    
    for row in comparison:
        print(f"{row[0]:20} | {row[1]:25} | {row[2]:20}")


def main():
    """
    Run all examples
    """
    print("FLEXIBLE CRYSTAL GRAPH VAE - USAGE EXAMPLES")
    print("This system automatically adapts to your dataset type!")
    
    example_with_extra_features()
    example_structure_only()
    example_automatic_detection()
    example_generation_and_analysis()
    example_integration_with_existing_code()
    comparison_table()
    
    print("\n" + "="*80)
    print("GETTING STARTED")
    print("="*80)
    
    print("\nFor your current coronene dimer dataset:")
    print("python flexible_vae_training.py ./data/MyData/ --feature-file ./data/MyData/features.csv")
    
    print("\nFor a structure-only dataset:")
    print("python flexible_vae_training.py ./data/SomeDataset/ --force-structure-only")
    
    print("\nFor automatic detection:")
    print("python flexible_vae_training.py ./data/SomeDataset/")
    
    print("\nThe system will automatically:")
    print("✓ Detect if you have extra features")
    print("✓ Create the appropriate VAE architecture") 
    print("✓ Train with the right loss functions")
    print("✓ Generate appropriate outputs")
    print("✓ Save compatible checkpoints")
    
    print("\nNo code changes needed - just run and the system adapts!")


if __name__ == '__main__':
    main()
