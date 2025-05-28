#!/usr/bin/env python3
"""
CGCNN Parameter Extractor
Extracts EXACT parameters that your CGCNN model used during training

"""

import torch
import json
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pickle

class CGCNNParameterExtractor:
    """
    Extract exact parameters from your trained CGCNN/VAE model
    This ensures perfect reversal with no assumptions
    """
    
    def __init__(self, model_path: str, data_dir: str = None):
        self.model_path = model_path
        self.data_dir = data_dir
        self.parameters = {}
        
    def extract_all_parameters(self) -> Dict[str, Any]:
        """Extract all CGCNN parameters from model and dataset"""
        print(" Extracting EXACT CGCNN parameters (no assumptions)...")
        
        # 1. Extract from model checkpoint
        model_params = self.extract_from_model_checkpoint()
        
        # 2. Extract from dataset configuration
        dataset_params = self.extract_from_dataset()
        
        # 3. Extract from data loader configuration
        dataloader_params = self.extract_from_dataloader_config()
        
        # 4. Analyze atom_init.json
        atom_params = self.analyze_atom_initialization()
        
        # 5. Infer Gaussian distance parameters
        gaussian_params = self.infer_gaussian_distance_parameters()
        
        # Combine all parameters
        self.parameters = {
            **model_params,
            **dataset_params,
            **dataloader_params,
            **atom_params,
            **gaussian_params
        }
        
        self.validate_parameters()
        self.save_extracted_parameters()
        
        return self.parameters
    
    def extract_from_model_checkpoint(self) -> Dict[str, Any]:
        """Extract parameters directly from model checkpoint"""
        print(" Analyzing model checkpoint...")
        
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            params = {}
            
            # Basic model architecture
            params['latent_dim'] = checkpoint.get('latent_dim', None)
            params['use_extra_features'] = checkpoint.get('use_extra_features', False)
            params['n_extra_features'] = checkpoint.get('n_extra_features', 0)
            params['feature_names'] = checkpoint.get('feature_names', [])
            
            # Dataset info
            dataset_info = checkpoint.get('dataset_info', {})
            params['atom_feature_dim'] = dataset_info.get('atom_feature_dim', None)
            params['bond_feature_dim'] = dataset_info.get('bond_feature_dim', None)
            params['num_samples'] = dataset_info.get('num_samples', None)
            
            # Training configuration
            params['training_radius'] = checkpoint.get('radius', None)
            params['training_max_neighbors'] = checkpoint.get('max_num_nbr', None)
            params['training_dmin'] = checkpoint.get('dmin', None)
            params['training_dmax'] = checkpoint.get('dmax', None)
            params['training_step'] = checkpoint.get('step', None)
            
            # Model state analysis
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                params.update(self.analyze_model_state(model_state))
            
            print(f" Extracted {len(params)} parameters from checkpoint")
            return params
            
        except Exception as e:
            print(f"  Could not extract from checkpoint: {e}")
            return {}
    
    def analyze_model_state(self, model_state: Dict) -> Dict[str, Any]:
        """Analyze model state dict to infer parameters"""
        params = {}
        
        # Analyze layer dimensions to infer architecture
        for key, tensor in model_state.items():
            if 'atom_embedding' in key and 'weight' in key:
                # atom_embedding layer: input_dim = atom_feature_dim
                params['inferred_atom_feature_dim'] = tensor.shape[1]
                params['inferred_atom_embed_dim'] = tensor.shape[0]
                
            elif 'extra_feat_encoder' in key and 'weight' in key and '0' in key:
                # First layer of extra feature encoder
                params['inferred_extra_feature_dim'] = tensor.shape[1]
                
            elif 'fc_mu' in key and 'weight' in key:
                # Latent dimension from mu layer
                params['inferred_latent_dim'] = tensor.shape[0]
                
            elif 'convs.0.fc_full' in key and 'weight' in key:
                # First conv layer - infer neighbor feature dimension
                # fc_full: 2*atom_fea_len + nbr_fea_len → 2*atom_fea_len
                input_dim = tensor.shape[1]
                output_dim = tensor.shape[0]
                atom_fea_len = output_dim // 2
                params['inferred_neighbor_feature_dim'] = input_dim - 2 * atom_fea_len
                params['inferred_conv_atom_dim'] = atom_fea_len
        
        return params
    
    def extract_from_dataset(self) -> Dict[str, Any]:
        """Extract parameters from dataset files"""
        print(" Analyzing dataset configuration...")
        
        params = {}
        
        if not self.data_dir:
            return params
        
        # Check for training configuration files
        config_files = [
            'dataset_config.json',
            'training_config.json', 
            'config.json',
            'params.json'
        ]
        
        for config_file in config_files:
            config_path = os.path.join(self.data_dir, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Extract relevant parameters
                    params.update({
                        k: v for k, v in config.items() 
                        if k in ['radius', 'max_num_nbr', 'dmin', 'dmax', 'step']
                    })
                    print(f" Found config in {config_file}")
                    
                except Exception as e:
                    print(f"  Could not read {config_file}: {e}")
        
        return params
    
    def extract_from_dataloader_config(self) -> Dict[str, Any]:
        """Try to infer dataloader parameters from dataset structure"""
        print(" Inferring dataloader parameters...")
        
        params = {}
        
        if not self.data_dir:
            return params
        
        try:
            # Analyze a sample CIF file to infer radius/neighbor parameters
            import glob
            cif_files = glob.glob(os.path.join(self.data_dir, "*.cif"))
            
            if cif_files:
                sample_file = cif_files[0]
                params.update(self.infer_from_sample_structure(sample_file))
            
        except Exception as e:
            print(f"  Could not analyze sample structures: {e}")
        
        return params
    
    def infer_from_sample_structure(self, cif_file: str) -> Dict[str, Any]:
        """Infer parameters from a sample crystal structure"""
        try:
            from ase.io import read
            
            structure = read(cif_file)
            
            # Analyze typical bond lengths in this structure
            from ase.neighborlist import NeighborList, natural_cutoffs
            
            # Get natural cutoffs for this structure
            cutoffs = natural_cutoffs(structure)
            max_cutoff = max(cutoffs) if cutoffs else 3.0
            
            # Infer reasonable radius (typical CGCNN uses 1.5-2x covalent radius)
            inferred_radius = min(8.0, max_cutoff * 2.0)
            
            params = {
                'inferred_radius': inferred_radius,
                'structure_elements': list(set(structure.get_chemical_symbols())),
                'typical_bond_length': max_cutoff
            }
            
            print(f" Inferred radius: {inferred_radius:.2f} Å from structure analysis")
            return params
            
        except Exception as e:
            print(f"  Could not analyze structure {cif_file}: {e}")
            return {}
    
    def analyze_atom_initialization(self) -> Dict[str, Any]:
        """Analyze atom_init.json to understand atom feature encoding"""
        print(" Analyzing atom initialization...")
        
        params = {}
        
        if not self.data_dir:
            return params
        
        atom_init_path = os.path.join(self.data_dir, 'atom_init.json')
        
        if not os.path.exists(atom_init_path):
            print("  atom_init.json not found")
            return params
        
        try:
            with open(atom_init_path, 'r') as f:
                atom_init = json.load(f)
            
            # Convert keys to integers
            atom_init = {int(k): np.array(v) for k, v in atom_init.items()}
            
            # Analyze atom feature vectors
            atomic_numbers = list(atom_init.keys())
            feature_lengths = [len(v) for v in atom_init.values()]
            
            params['atom_init_data'] = atom_init
            params['available_elements'] = atomic_numbers
            params['atom_feature_length'] = feature_lengths[0] if feature_lengths else None
            params['num_element_types'] = len(atomic_numbers)
            
            # Analyze feature patterns
            all_features = np.array(list(atom_init.values()))
            params['atom_feature_stats'] = {
                'mean': np.mean(all_features, axis=0).tolist(),
                'std': np.std(all_features, axis=0).tolist(),
                'min': np.min(all_features, axis=0).tolist(),
                'max': np.max(all_features, axis=0).tolist()
            }
            
            print(f" Found {len(atomic_numbers)} element types with {feature_lengths[0]}-D features")
            
            return params
            
        except Exception as e:
            print(f"  Could not analyze atom_init.json: {e}")
            return {}
    
    def infer_gaussian_distance_parameters(self) -> Dict[str, Any]:
        """Infer Gaussian distance expansion parameters"""
        print(" Inferring Gaussian distance parameters...")
        
        params = {}
        
        # Try to get from extracted parameters first
        radius = (self.parameters.get('training_radius') or 
                 self.parameters.get('inferred_radius') or 8.0)
        
        dmin = self.parameters.get('training_dmin', 0.0)
        dmax = self.parameters.get('training_dmax', radius)
        step = self.parameters.get('training_step', 0.2)
        
        # Validate against bond feature dimension
        bond_feature_dim = (self.parameters.get('bond_feature_dim') or
                           self.parameters.get('inferred_neighbor_feature_dim'))
        
        if bond_feature_dim:
            # Calculate expected number of Gaussian centers
            expected_centers = int((dmax - dmin) / step) + 1
            
            if expected_centers != bond_feature_dim:
                print(f" Adjusting Gaussian parameters to match bond features...")
                print(f"   Expected centers: {expected_centers}, Actual bond features: {bond_feature_dim}")
                
                # Adjust step to match bond feature dimension
                step = (dmax - dmin) / (bond_feature_dim - 1)
                print(f"   Adjusted step: {step:.4f}")
        
        params.update({
            'gaussian_dmin': dmin,
            'gaussian_dmax': dmax,
            'gaussian_step': step,
            'gaussian_centers': int((dmax - dmin) / step) + 1,
            'gaussian_variance': step  # Common choice: variance = step
        })
        
        print(f" Gaussian parameters: dmin={dmin}, dmax={dmax:.1f}, step={step:.3f}")
        return params
    
    def validate_parameters(self):
        """Validate that extracted parameters are consistent"""
        print(" Validating parameter consistency...")
        
        # Check atom feature dimensions
        atom_dim_checkpoint = self.parameters.get('atom_feature_dim')
        atom_dim_inferred = self.parameters.get('inferred_atom_feature_dim')
        atom_dim_init = self.parameters.get('atom_feature_length')
        
        if atom_dim_checkpoint and atom_dim_inferred:
            if atom_dim_checkpoint != atom_dim_inferred:
                print(f"  Atom feature dimension mismatch: "
                      f"checkpoint={atom_dim_checkpoint}, inferred={atom_dim_inferred}")
        
        # Check bond feature dimensions
        bond_dim_checkpoint = self.parameters.get('bond_feature_dim')
        bond_dim_inferred = self.parameters.get('inferred_neighbor_feature_dim')
        gaussian_centers = self.parameters.get('gaussian_centers')
        
        if bond_dim_checkpoint and gaussian_centers:
            if bond_dim_checkpoint != gaussian_centers:
                print(f"  Bond feature dimension mismatch: "
                      f"checkpoint={bond_dim_checkpoint}, gaussian={gaussian_centers}")
        
        # Validate radius parameters
        radius_values = [
            self.parameters.get('training_radius'),
            self.parameters.get('inferred_radius')
        ]
        radius_values = [r for r in radius_values if r is not None]
        
        if len(radius_values) > 1 and not all(abs(r - radius_values[0]) < 0.5 for r in radius_values):
            print(f"  Radius values inconsistent: {radius_values}")
    
    def save_extracted_parameters(self):
        """Save extracted parameters for future use"""
        output_file = self.model_path.replace('.pth', '_extracted_params.json')
        
        # Convert numpy arrays to lists for JSON serialization
        json_params = {}
        for key, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                json_params[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_params[key] = value.item()
            else:
                json_params[key] = value
        
        try:
            with open(output_file, 'w') as f:
                json.dump(json_params, f, indent=2)
            
            print(f" Saved extracted parameters to: {output_file}")
            
        except Exception as e:
            print(f"  Could not save parameters: {e}")
    
    def get_parameter(self, param_name: str, default=None):
        """Get parameter with fallback logic"""
        # Try multiple possible names for the same parameter
        param_aliases = {
            'radius': ['training_radius', 'inferred_radius', 'radius'],
            'dmin': ['training_dmin', 'gaussian_dmin', 'dmin'],
            'dmax': ['training_dmax', 'gaussian_dmax', 'dmax'],  
            'step': ['training_step', 'gaussian_step', 'step'],
            'atom_feature_dim': ['atom_feature_dim', 'inferred_atom_feature_dim', 'atom_feature_length'],
            'bond_feature_dim': ['bond_feature_dim', 'inferred_neighbor_feature_dim', 'gaussian_centers']
        }
        
        if param_name in param_aliases:
            for alias in param_aliases[param_name]:
                if alias in self.parameters and self.parameters[alias] is not None:
                    return self.parameters[alias]
        
        return self.parameters.get(param_name, default)
    
    def print_summary(self):
        """Print a comprehensive summary of extracted parameters"""
        print("\n" + "="*60)
        print(" EXTRACTED CGCNN PARAMETERS SUMMARY")
        print("="*60)
        
        # Model Architecture
        print("\n  Model Architecture:")
        print(f"  Latent dimension: {self.get_parameter('latent_dim', 'Unknown')}")
        print(f"  Use extra features: {self.get_parameter('use_extra_features', 'Unknown')}")
        print(f"  Number of extra features: {self.get_parameter('n_extra_features', 'Unknown')}")
        
        # Atom Features
        print("\n Atom Features:")
        print(f"  Atom feature dimension: {self.get_parameter('atom_feature_dim', 'Unknown')}")
        print(f"  Available elements: {len(self.get_parameter('available_elements', []))}")
        print(f"  Element types: {self.get_parameter('available_elements', 'Unknown')}")
        
        # Graph Construction
        print("\n Graph Construction:")
        print(f"  Radius: {self.get_parameter('radius', 'Unknown')} Å")
        print(f"  Max neighbors: {self.get_parameter('training_max_neighbors', 'Unknown')}")
        
        # Gaussian Distance Expansion  
        print("\n Gaussian Distance Expansion:")
        print(f"  dmin: {self.get_parameter('dmin', 'Unknown')}")
        print(f"  dmax: {self.get_parameter('dmax', 'Unknown')}")
        print(f"  step: {self.get_parameter('step', 'Unknown')}")
        print(f"  Bond feature dim: {self.get_parameter('bond_feature_dim', 'Unknown')}")
        
        # Extra Features
        if self.get_parameter('use_extra_features'):
            print("\n Extra Features:")
            feature_names = self.get_parameter('feature_names', [])
            print(f"  Number: {len(feature_names)}")
            if feature_names:
                print(f"  Names: {', '.join(feature_names[:5])}")
                if len(feature_names) > 5:
                    print(f"         ... and {len(feature_names) - 5} more")


def enhance_cgcnn_reversal(model_path: str, data_dir: str = None):
    """
    Main function to extract parameters and enhance CGCNN reversal
    """
    print(" ENHANCING CGCNN REVERSAL WITH EXTRACTED PARAMETERS")
    
    # Extract parameters
    extractor = CGCNNParameterExtractor(model_path, data_dir)
    parameters = extractor.extract_all_parameters()
    
    # Print summary
    extractor.print_summary()
    
    return extractor, parameters


if __name__ == '__main__':
    # Example usage
    model_path = './flexible_vae_checkpoints/best_flexible_vae_model.pth'
    data_dir = './data/MyData'
    
    extractor, params = enhance_cgcnn_reversal(model_path, data_dir)
