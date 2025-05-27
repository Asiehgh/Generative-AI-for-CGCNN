import torch
import numpy as np
import os
import csv
import warnings
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset


def flexible_collate_pool(dataset_list):
    """
    Flexible collate function that handles datasets with or without extra features
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_extra_fea = []
    batch_cif_ids = []
    base_idx = 0
    has_extra_features = False
    
    for i, data in enumerate(dataset_list):
        if len(data) == 3:  # (input_tuple, target, cif_id)
            input_tuple, target, cif_id = data
            
            if len(input_tuple) == 5:  # Has extra features
                atom_fea, nbr_fea, nbr_fea_idx, extra_fea = input_tuple[:4]
                has_extra_features = True
                batch_extra_fea.append(extra_fea)
            elif len(input_tuple) == 4:  # No extra features
                atom_fea, nbr_fea, nbr_fea_idx = input_tuple[:3]
            else:
                raise ValueError(f"Unexpected input tuple length: {len(input_tuple)}")
        else:
            raise ValueError(f"Unexpected data format: {len(data)} elements")
        
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    
    # Create the output tuple
    output_inputs = [
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        crystal_atom_idx
    ]
    
    # Add extra features if present
    if has_extra_features and batch_extra_fea:
        output_inputs.append(torch.stack(batch_extra_fea, dim=0))
    
    return (tuple(output_inputs),
            torch.stack(batch_target, dim=0),
            batch_cif_ids)


class FlexibleCIFData(Dataset):
    """
    Flexible CIF dataset that can work with or without extra features
    """
    
    def __init__(self, root_dir, feature_file=None, max_num_nbr=12, radius=8, 
                 dmin=0, step=0.2, random_seed=123, use_extra_features=None):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.feature_file = feature_file
        
        # Auto-detect if we should use extra features
        if use_extra_features is None:
            self.use_extra_features = feature_file is not None and os.path.exists(feature_file)
        else:
            self.use_extra_features = use_extra_features
            
        print(f"Flexible CIF Dataset initialized:")
        print(f"  Root directory: {root_dir}")
        print(f"  Feature file: {feature_file}")
        print(f"  Use extra features: {self.use_extra_features}")
        
        # Load and verify id_prop.csv
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        if not os.path.exists(id_prop_file):
            raise FileNotFoundError('id_prop.csv does not exist!')
            
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Initialize feature processing
        self.extra_features = {}
        self.feature_normalizer = None
        self.feature_names = []
        
        # Load and process extra features if enabled
        if self.use_extra_features:
            self._load_extra_features(feature_file)
            self._compute_feature_normalization()
        else:
            print("Operating in structure-only mode (no extra features)")

        # Shuffle data
        import random
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
        # Initialize atom features and distance functions
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        if not os.path.exists(atom_init_file):
            raise FileNotFoundError('atom_init.json does not exist!')
            
        from data import AtomCustomJSONInitializer, GaussianDistance
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def _load_extra_features(self, feature_file):
        """Load extra features from CSV file - handles ANY number of features"""
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f'Feature file {feature_file} does not exist!')
            
        print(f"Loading extra features from {feature_file}...")
        
        with open(feature_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            self.feature_names = header[1:]  # Skip ID column
            num_features = len(self.feature_names)
            
            print(f"Detected {num_features} features:")
            
            # Print features in organized way based on count
            if num_features <= 10:
                print(f"  Features: {', '.join(self.feature_names)}")
            else:
                print(f"  First 10: {', '.join(self.feature_names[:10])}")
                if num_features > 10:
                    print(f"  ... and {num_features - 10} more")
            
            # Provide feature count context
            if num_features <= 5:
                print(f"  → Small feature set: Using lightweight architecture")
            elif num_features <= 20:
                print(f"  → Medium feature set: Using moderate architecture")
            elif num_features <= 50:
                print(f"  → Large feature set: Using expanded architecture")
            else:
                print(f"  → Very large feature set: Using full-capacity architecture")
            
            # Initialize default values
            self.default_values = {name: 0.0 for name in self.feature_names}
            
            # Read data
            loaded_count = 0
            for row_idx, row in enumerate(reader):
                try:
                    crystal_id = row[0]
                    features = [float(val) for val in row[1:]]
                    
                    if len(features) != num_features:
                        print(f"Warning: Row {row_idx + 2} has {len(features)} features, expected {num_features}")
                        # Pad or truncate to match expected length
                        if len(features) < num_features:
                            features.extend([0.0] * (num_features - len(features)))
                            print(f"  → Padded with zeros to {num_features} features")
                        else:
                            features = features[:num_features]
                            print(f"  → Truncated to {num_features} features")
                    
                    self.extra_features[crystal_id] = features
                    loaded_count += 1
                    
                    # Print first few entries (adapted for any number of features)
                    if loaded_count <= 3:
                        if num_features <= 8:
                            feature_str = ", ".join([f"{self.feature_names[i]}: {val:.4f}" 
                                                   for i, val in enumerate(features)])
                        else:
                            feature_str = ", ".join([f"{self.feature_names[i]}: {val:.4f}" 
                                                   for i, val in enumerate(features[:6])])
                            feature_str += f", ... (+{num_features - 6} more)"
                        
                        print(f"  Sample {loaded_count} - {crystal_id}: {feature_str}")
                        
                except (IndexError, ValueError) as e:
                    print(f"Error processing row {row_idx + 2}: {row} - {e}")
                    continue
                    
        print(f"Successfully loaded {loaded_count} crystals with {num_features} features each")
        
        # Validate feature loading
        if loaded_count == 0:
            raise ValueError("No valid features were loaded from the file!")
        
        # Print enhanced feature statistics
        self._print_feature_statistics()

    def _print_feature_statistics(self):
        """Print comprehensive feature statistics adapted for any number of features"""
        if not self.extra_features:
            return
            
        num_features = len(self.feature_names)
        print(f"\nFeature Statistics Summary ({num_features} features):")
        print("-" * 60)
        
        # Calculate statistics for all features
        feature_stats = {}
        for i, name in enumerate(self.feature_names):
            values = [features[i] for features in self.extra_features.values()]
            feature_stats[name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'range': max(values) - min(values)
            }
        
        # Print statistics in organized format
        if num_features <= 15:
            # Print all features for small sets
            for name, stats in feature_stats.items():
                print(f"{name:25} | Min: {stats['min']:8.4f} | Max: {stats['max']:8.4f} | "
                      f"Mean: {stats['mean']:8.4f} | Std: {stats['std']:8.4f}")
        else:
            # Print summary for large feature sets
            print(f"{'Feature Name':25} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8}")
            print("-" * 75)
            
            # Show first 10 features
            for i, (name, stats) in enumerate(list(feature_stats.items())[:10]):
                print(f"{name:25} | {stats['min']:8.4f} | {stats['max']:8.4f} | "
                      f"{stats['mean']:8.4f} | {stats['std']:8.4f}")
            
            if num_features > 10:
                print(f"... and {num_features - 10} more features")
                
                # Show summary statistics
                all_mins = [stats['min'] for stats in feature_stats.values()]
                all_maxs = [stats['max'] for stats in feature_stats.values()]
                all_means = [stats['mean'] for stats in feature_stats.values()]
                all_stds = [stats['std'] for stats in feature_stats.values()]
                
                print(f"\nOverall Statistics Across All {num_features} Features:")
                print(f"  Minimum values range: {min(all_mins):.4f} to {max(all_mins):.4f}")
                print(f"  Maximum values range: {min(all_maxs):.4f} to {max(all_maxs):.4f}")
                print(f"  Mean values range: {min(all_means):.4f} to {max(all_means):.4f}")
                print(f"  Std dev range: {min(all_stds):.4f} to {max(all_stds):.4f}")
        
        # Feature diversity analysis
        ranges = [stats['range'] for stats in feature_stats.values()]
        large_range_features = [name for name, stats in feature_stats.items() if stats['range'] > np.percentile(ranges, 75)]
        
        if large_range_features:
            print(f"\nFeatures with large value ranges (top 25%):")
            for name in large_range_features[:5]:  # Show top 5
                range_val = feature_stats[name]['range']
                print(f"  {name}: range = {range_val:.4f}")
            if len(large_range_features) > 5:
                print(f"  ... and {len(large_range_features) - 5} others")

    def _compute_feature_normalization(self):
        """Compute normalization statistics for all features (any number)"""
        if not self.extra_features:
            return

        num_features = len(self.feature_names)
        print(f"\nComputing normalization for {num_features} features...")

        # Initialize lists for each feature
        feature_values = {name: [] for name in self.feature_names}
        
        # Collect all values for each feature
        for features in self.extra_features.values():
            for i, value in enumerate(features):
                if i < len(self.feature_names):  # Safety check
                    feature_values[self.feature_names[i]].append(value)

        # Compute statistics for each feature
        self.feature_normalizer = {}
        zero_std_features = []
        
        for name, values in feature_values.items():
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                if std_val == 0:
                    std_val = 1.0  # Prevent division by zero
                    zero_std_features.append(name)
                
                self.feature_normalizer[name] = {
                    'mean': mean_val,
                    'std': std_val
                }
        
        if zero_std_features:
            print(f"Warning: {len(zero_std_features)} features have zero std deviation:")
            for name in zero_std_features[:5]:  # Show first 5
                print(f"  {name}")
            if len(zero_std_features) > 5:
                print(f"  ... and {len(zero_std_features) - 5} others")
        
        print(f"Normalization computed for {len(self.feature_normalizer)} features")

    def _normalize_features(self, features):
        """Normalize features using computed statistics"""
        if self.feature_normalizer is None:
            return torch.tensor(features, dtype=torch.float32)

        normalized_features = []
        for i, feature in enumerate(features):
            name = self.feature_names[i]
            norm_value = (feature - self.feature_normalizer[name]['mean']) / \
                        self.feature_normalizer[name]['std']
            normalized_features.append(norm_value)
        
        return torch.tensor(normalized_features, dtype=torch.float32)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        
        # Load crystal structure
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))
        
        # Process atom features
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                             for i in range(len(crystal))])
        
        # Process neighbor features
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} has fewer than {self.max_num_nbr} neighbors')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                 [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                             [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        
        # Convert to tensors
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(self.gdf.expand(np.array(nbr_fea)))
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        
        # Create output tuple based on whether we use extra features
        if self.use_extra_features:
            # Get and normalize extra features
            if cif_id in self.extra_features:
                features = self.extra_features[cif_id]
            else:
                warnings.warn(f"No extra features found for crystal {cif_id}. Using default values.")
                features = [self.default_values[name] for name in self.feature_names]
            
            extra_fea = self._normalize_features(features)
            return (atom_fea, nbr_fea, nbr_fea_idx, extra_fea), target, cif_id
        else:
            # Structure-only mode
            return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


class StructureOnlyCIFData(Dataset):
    """
    Simplified CIF dataset for structure-only (no extra features) use cases
    """
    
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2, random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        
        print(f"Structure-only CIF Dataset initialized:")
        print(f"  Root directory: {root_dir}")
        print(f"  No extra features will be used")
        
        # Load id_prop.csv
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        if not os.path.exists(id_prop_file):
            raise FileNotFoundError('id_prop.csv does not exist!')
            
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Shuffle data
        import random
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        
        # Initialize atom features and distance functions
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        if not os.path.exists(atom_init_file):
            raise FileNotFoundError('atom_init.json does not exist!')
            
        from data import AtomCustomJSONInitializer, GaussianDistance
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        
        # Set empty feature names for compatibility
        self.feature_names = []
        self.use_extra_features = False

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        
        # Load crystal structure
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id + '.cif'))
        
        # Process atom features
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                             for i in range(len(crystal))])
        
        # Process neighbor features
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(f'{cif_id} has fewer than {self.max_num_nbr} neighbors')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                 [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                             [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        
        # Convert to tensors
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(self.gdf.expand(np.array(nbr_fea)))
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        
        # Return structure-only format
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


def create_flexible_dataset(root_dir, feature_file=None, max_num_nbr=12, radius=8, 
                           dmin=0, step=0.2, random_seed=123, force_structure_only=False):
    """
    Factory function to create appropriate dataset based on available files
    """
    print("Creating flexible dataset...")
    
    # Check if forcing structure-only mode
    if force_structure_only:
        print("Forced structure-only mode")
        return StructureOnlyCIFData(
            root_dir=root_dir,
            max_num_nbr=max_num_nbr,
            radius=radius,
            dmin=dmin,
            step=step,
            random_seed=random_seed
        )
    
    # Auto-detect mode based on available files
    has_feature_file = feature_file is not None and os.path.exists(feature_file)
    
    if has_feature_file:
        print("Feature file found - using flexible dataset with extra features")
        return FlexibleCIFData(
            root_dir=root_dir,
            feature_file=feature_file,
            max_num_nbr=max_num_nbr,
            radius=radius,
            dmin=dmin,
            step=step,
            random_seed=random_seed,
            use_extra_features=True
        )
    else:
        print("No feature file found - using structure-only dataset")
        return StructureOnlyCIFData(
            root_dir=root_dir,
            max_num_nbr=max_num_nbr,
            radius=radius,
            dmin=dmin,
            step=step,
            random_seed=random_seed
        )


def get_dataset_info(dataset):
    """
    Get information about the dataset type and dimensions
    """
    sample_data, target, cif_id = dataset[0]
    
    info = {
        'dataset_type': type(dataset).__name__,
        'has_extra_features': hasattr(dataset, 'use_extra_features') and dataset.use_extra_features,
        'num_samples': len(dataset),
        'atom_feature_dim': sample_data[0].shape[-1],
        'bond_feature_dim': sample_data[1].shape[-1],
        'sample_cif_id': cif_id
    }
    
    if info['has_extra_features']:
        info['num_extra_features'] = len(dataset.feature_names)
        info['feature_names'] = dataset.feature_names
        info['extra_feature_dim'] = sample_data[3].shape[-1] if len(sample_data) > 3 else 0
    else:
        info['num_extra_features'] = 0
        info['feature_names'] = []
        info['extra_feature_dim'] = 0
    
    return info


# Example usage and testing functions
def test_flexible_dataset(root_dir, feature_file=None):
    """
    Test function to verify flexible dataset functionality
    """
    print("Testing flexible dataset functionality...")
    
    # Test with extra features (if available)
    if feature_file and os.path.exists(feature_file):
        print("\n1. Testing with extra features:")
        dataset_with_features = create_flexible_dataset(
            root_dir=root_dir,
            feature_file=feature_file
        )
        
        info = get_dataset_info(dataset_with_features)
        print(f"Dataset info: {info}")
        
        # Test data loading
        sample_data, target, cif_id = dataset_with_features[0]
        print(f"Sample data structure: {len(sample_data)} components")
        print(f"Atom features shape: {sample_data[0].shape}")
        print(f"Bond features shape: {sample_data[1].shape}")
        if len(sample_data) > 3:
            print(f"Extra features shape: {sample_data[3].shape}")
        print(f"Target: {target.item()}")
        print(f"CIF ID: {cif_id}")
    
    # Test structure-only mode
    print("\n2. Testing structure-only mode:")
    dataset_structure_only = create_flexible_dataset(
        root_dir=root_dir,
        force_structure_only=True
    )
    
    info = get_dataset_info(dataset_structure_only)
    print(f"Dataset info: {info}")
    
    # Test data loading
    sample_data, target, cif_id = dataset_structure_only[0]
    print(f"Sample data structure: {len(sample_data)} components")
    print(f"Atom features shape: {sample_data[0].shape}")
    print(f"Bond features shape: {sample_data[1].shape}")
    print(f"Target: {target.item()}")
    print(f"CIF ID: {cif_id}")
    
    # Test collate function
    print("\n3. Testing collate function:")
    from torch.utils.data import DataLoader
    
    # Test with structure-only
    loader = DataLoader(dataset_structure_only, batch_size=2, collate_fn=flexible_collate_pool)
    batch = next(iter(loader))
    inputs, targets, cif_ids = batch
    print(f"Structure-only batch - Inputs: {len(inputs)} components")
    print(f"Batch atom features: {inputs[0].shape}")
    print(f"Batch targets: {targets.shape}")
    
    # Test with extra features (if available)
    if feature_file and os.path.exists(feature_file):
        loader_with_features = DataLoader(dataset_with_features, batch_size=2, collate_fn=flexible_collate_pool)
        batch = next(iter(loader_with_features))
        inputs, targets, cif_ids = batch
        print(f"With-features batch - Inputs: {len(inputs)} components")
        print(f"Batch atom features: {inputs[0].shape}")
        if len(inputs) > 4:
            print(f"Batch extra features: {inputs[4].shape}")
        print(f"Batch targets: {targets.shape}")
    
    print("\nFlexible dataset testing completed successfully!")


if __name__ == '__main__':
    # Example usage
    root_dir = './data/MyData/'
    feature_file = './data/MyData/features.csv'
    
    test_flexible_dataset(root_dir, feature_file)
