#!/usr/bin/env python3
"""
Flexible Crystal Graph VAE - Core Model Implementation
Automatically adapts to any number of features (1, 5, 12, 50, 100+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import ConvLayer  # Your existing CGCNN ConvLayer
from data import collate_pool
from balanced_vae_model import balanced_vae_loss_function

class FlexibleCrystalGraphVAE(nn.Module):
    """
    Ultra-Flexible Variational Autoencoder for Crystal Graph Convolutional Networks
    Works with or without extra features - automatically adapts to ANY number of features
    """
    
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, 
                 n_conv=3, h_fea_len=128, n_extra_features=None, 
                 latent_dim=64, max_atoms=200, use_extra_features=True,
                 adaptive_feature_scaling=True):
        super(FlexibleCrystalGraphVAE, self).__init__()
        
        self.atom_fea_len = atom_fea_len
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.n_extra_features = n_extra_features if n_extra_features is not None else 0
        self.use_extra_features = use_extra_features and (self.n_extra_features > 0)
        self.orig_atom_fea_len = orig_atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.adaptive_feature_scaling = adaptive_feature_scaling
        
        print(f"Initializing Ultra-Flexible VAE:")
        print(f"  Use extra features: {self.use_extra_features}")
        print(f"  Number of extra features: {self.n_extra_features}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Adaptive feature scaling: {adaptive_feature_scaling}")
        
        # ============= ENCODER =============
        # Crystal structure encoder (based on your CGCNN)
        self.atom_embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        
        # Extra features encoder (only if using extra features)
        # ADAPTIVE ARCHITECTURE: Scales with number of features
        if self.use_extra_features:
            # Dynamically size the feature processing network based on number of features
            if self.adaptive_feature_scaling:
                # Scale intermediate dimensions based on number of features
                if self.n_extra_features <= 5:
                    # Small feature sets: simpler network
                    extra_hidden_dim = max(16, h_fea_len // 8)
                elif self.n_extra_features <= 20:
                    # Medium feature sets: moderate network
                    extra_hidden_dim = max(32, h_fea_len // 4)
                elif self.n_extra_features <= 50:
                    # Large feature sets: larger network
                    extra_hidden_dim = max(64, h_fea_len // 2)
                else:
                    # Very large feature sets: full capacity
                    extra_hidden_dim = h_fea_len
                    
                print(f"  Extra features hidden dim: {extra_hidden_dim} (adaptive scaling)")
            else:
                # Fixed scaling
                extra_hidden_dim = h_fea_len // 4
                print(f"  Extra features hidden dim: {extra_hidden_dim} (fixed scaling)")
            
            # Build feature encoder with appropriate capacity
            layers = []
            
            # First layer: features -> hidden
            layers.append(nn.Linear(self.n_extra_features, extra_hidden_dim))
            layers.append(nn.BatchNorm1d(extra_hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            
            # Add additional layers for large feature sets
            if self.n_extra_features > 20:
                layers.append(nn.Linear(extra_hidden_dim, extra_hidden_dim))
                layers.append(nn.BatchNorm1d(extra_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
            
            if self.n_extra_features > 50:
                layers.append(nn.Linear(extra_hidden_dim, extra_hidden_dim))
                layers.append(nn.BatchNorm1d(extra_hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
            
            self.extra_feat_encoder = nn.Sequential(*layers)
            self.extra_hidden_dim = extra_hidden_dim
            combined_input_dim = atom_fea_len + extra_hidden_dim
        else:
            self.extra_feat_encoder = None
            self.extra_hidden_dim = 0
            combined_input_dim = atom_fea_len
        
        # Combined feature processing
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_input_dim, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent space mappings (mean and log variance)
        self.fc_mu = nn.Linear(h_fea_len, latent_dim)
        self.fc_logvar = nn.Linear(h_fea_len, latent_dim)
        
        # ============= DECODER =============
        # Latent to features
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Extra features decoder (only if using extra features)
        # ADAPTIVE DECODER: Scales with number of features
        if self.use_extra_features:
            decoder_layers = []
            
            # Start from latent representation
            current_dim = h_fea_len // 2 if self.use_extra_features else h_fea_len
            
            # Add layers based on feature complexity
            if self.n_extra_features > 50:
                # Very large feature sets: deep decoder
                decoder_layers.extend([
                    nn.Linear(current_dim, self.extra_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(self.extra_hidden_dim, self.extra_hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(self.extra_hidden_dim // 2, self.n_extra_features)
                ])
            elif self.n_extra_features > 20:
                # Large feature sets: medium decoder
                decoder_layers.extend([
                    nn.Linear(current_dim, self.extra_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.extra_hidden_dim, self.n_extra_features)
                ])
            else:
                # Small to medium feature sets: simple decoder
                decoder_layers.extend([
                    nn.Linear(current_dim, max(self.extra_hidden_dim, self.n_extra_features * 2)),
                    nn.ReLU(),
                    nn.Linear(max(self.extra_hidden_dim, self.n_extra_features * 2), self.n_extra_features)
                ])
            
            self.extra_feat_decoder = nn.Sequential(*decoder_layers)
            print(f"  Feature decoder depth: {len([l for l in decoder_layers if isinstance(l, nn.Linear)])} layers")
        else:
            self.extra_feat_decoder = None
        
        # Atom features decoder
        struct_decoder_input = h_fea_len // 2 if self.use_extra_features else h_fea_len
        self.atom_feat_decoder = nn.Sequential(
            nn.Linear(struct_decoder_input + 1, atom_fea_len * 2),
            nn.ReLU(),
            nn.Linear(atom_fea_len * 2, atom_fea_len),
            nn.ReLU()
        )
        
        # Number of atoms predictor
        self.num_atoms_predictor = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.ReLU(),
            nn.Linear(h_fea_len // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1, will be scaled
        )
        
        # Atom type decoder (predicts atom type probabilities)
        self.atom_type_decoder = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, orig_atom_fea_len),
            nn.Softmax(dim=-1)
        )
        
        # Bond feature decoder
        self.bond_feat_decoder = nn.Sequential(
            nn.Linear(atom_fea_len * 2, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, nbr_fea_len),
            nn.Sigmoid()
        )
        
    def encode(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea=None):
        atom_fea = self.atom_embedding(atom_fea)        
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        crystal_fea = self.pooling(atom_fea, crystal_atom_idx)

        if self.use_extra_features and extra_fea is not None:
            extra_fea_encoded = self.extra_feat_encoder(extra_fea)

            combined_fea = torch.cat([crystal_fea, extra_fea_encoded], dim=1)

        else:
            combined_fea = crystal_fea

            
      
        combined_fea = self.feature_combiner(combined_fea)
        mu = self.fc_mu(combined_fea)
        logvar = self.fc_logvar(combined_fea)        
        return mu, logvar
        

       
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, num_atoms=None):
        """
        Decode latent vector to crystal structure (and optionally extra features)
        """
        batch_size = z.size(0)
        
        # Decode latent to high-level features
        decoded_features = self.latent_to_features(z)
        
        # Split features for different outputs
        if self.use_extra_features:
            feat_dim = decoded_features.size(1) // 2
            extra_feat_part = decoded_features[:, :feat_dim]
            struct_feat_part = decoded_features[:, feat_dim:]
            
            # Decode extra features
            reconstructed_extra_fea = self.extra_feat_decoder(extra_feat_part)
        else:
            struct_feat_part = decoded_features
            reconstructed_extra_fea = None
        
        # Predict number of atoms if not provided
        if num_atoms is None:
            num_atoms_raw = self.num_atoms_predictor(decoded_features)
            num_atoms = (num_atoms_raw * self.max_atoms).round().long()
            num_atoms = torch.clamp(num_atoms, min=1, max=self.max_atoms)
        else:
            if isinstance(num_atoms, int):
                num_atoms = torch.full((batch_size, 1), num_atoms, device=z.device).long()
        
        # Generate atom features for each crystal
        max_atoms_in_batch = num_atoms.max().item()
        
        # Expand structural features for each atom
        expanded_struct_feat = struct_feat_part.unsqueeze(1).expand(
            batch_size, max_atoms_in_batch, -1
        )
        
        # Add positional encoding (simple version)
        pos_encoding = torch.arange(max_atoms_in_batch, device=z.device).float()
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1)
        pos_encoding = pos_encoding.unsqueeze(2) / max_atoms_in_batch
        
        # Concatenate with expanded features
        if expanded_struct_feat.size(2) > 1:
            expanded_struct_feat = torch.cat([
                expanded_struct_feat, 
                pos_encoding.expand(-1, -1, 1)
            ], dim=2)


       
        # Decode to atom features
        atom_features = self.atom_feat_decoder(expanded_struct_feat.view(-1, expanded_struct_feat.size(2)))
        atom_features = atom_features.view(batch_size, max_atoms_in_batch, -1)
        
        # Decode atom types
        atom_types = self.atom_type_decoder(atom_features.view(-1, atom_features.size(2)))
        atom_types = atom_types.view(batch_size, max_atoms_in_batch, -1)
        
        # Generate bond features (simplified - assumes full connectivity)
        bond_features_list = []
        for i in range(batch_size):
            n_atoms = num_atoms[i].item()
            if n_atoms > 1:
                # Create pairwise atom features for bond prediction
                atom_pairs = []
                for j in range(n_atoms):
                    for k in range(n_atoms):
                        if j != k:
                            pair_feat = torch.cat([
                                atom_features[i, j], 
                                atom_features[i, k]
                            ])
                            atom_pairs.append(pair_feat)
                
                if atom_pairs:
                    atom_pairs = torch.stack(atom_pairs)
                    bond_feats = self.bond_feat_decoder(atom_pairs)
                    bond_features_list.append(bond_feats)
                else:
                    bond_features_list.append(torch.zeros(1, self.nbr_fea_len, device=z.device))
            else:
                bond_features_list.append(torch.zeros(1, self.nbr_fea_len, device=z.device))
        
        result = {
            'atom_features': atom_features,
            'atom_types': atom_types,
            'num_atoms': num_atoms,
            'bond_features': bond_features_list
        }
        
        # Only include extra features if they're being used
        if self.use_extra_features:
            result['extra_features'] = reconstructed_extra_fea
        
        return result
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea=None):
        """
        Forward pass through VAE - automatically handles presence/absence of extra features
        """
        # Ensure inputs are float32
        atom_fea = atom_fea.float()
        nbr_fea = nbr_fea.float()
        if extra_fea is not None:
            extra_fea = extra_fea.float()
        
        # Encode
        mu, logvar = self.encode(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z)
        
        return {
            'reconstructed': reconstructed,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def generate(self, num_samples=1, num_atoms=None, device='cpu'):
        """
        Generate new crystal structures (with or without extra features)
        """
        self.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode to crystal structures
            generated = self.decode(z, num_atoms=num_atoms)
            
        return generated
    
    def interpolate(self, start_structure, end_structure, num_steps=10):
        """
        Interpolate between two crystal structures in latent space
        """
        self.eval()
        with torch.no_grad():
            # Encode both structures
            mu1, _ = self.encode(*start_structure)
            mu2, _ = self.encode(*end_structure)
            
            # Create interpolation path
            alphas = torch.linspace(0, 1, num_steps, device=mu1.device)
            interpolated_structures = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                interpolated = self.decode(z_interp)
                interpolated_structures.append(interpolated)
            
        return interpolated_structures
    

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        pooled = torch.cat(summed_fea, dim=0)
        if not hasattr(self, 'pool_transform'):
            self.pool_transform = nn.Linear(pooled.size(1), self.atom_fea_len).to(pooled.device)
    
        return self.pool_transform(pooled)

def flexible_vae_loss_function(reconstructed, original, mu, logvar, beta=1.0, use_extra_features=True):
    """
    Flexible VAE loss function that works with or without extra features
    """
    batch_size = mu.size(0)
    
    # Reconstruction losses
    reconstruction_loss = 0
    loss_components = {}
    
    # Extra features reconstruction loss (only if using extra features)
    if use_extra_features and 'extra_features' in original and 'extra_features' in reconstructed:
        extra_feat_loss = F.mse_loss(
            reconstructed['extra_features'], 
            original['extra_features'], 
            reduction='sum'
        )
        reconstruction_loss += extra_feat_loss
        loss_components['extra_features_loss'] = extra_feat_loss.item()
    
    # Atom features reconstruction loss (always present)
    #if 'atom_features' in original and 'atom_features' in reconstructed:
    #    atom_feat_loss = 0
    #    for i in range(batch_size):
    #        orig_atoms = original['atom_features'][i]
    #        recon_atoms = reconstructed['atom_features'][i]
    #        min_atoms = min(orig_atoms.size(0), recon_atoms.size(0))
    #        
    #        if min_atoms > 0:
    #            atom_feat_loss += F.mse_loss(
    #                recon_atoms[:min_atoms], 
    #                orig_atoms[:min_atoms], 
    #                reduction='sum'
    #            )
    #    reconstruction_loss += atom_feat_loss
    #    loss_components['atom_features_loss'] = atom_feat_loss

    # Atom features reconstruction loss (FLEXIBLE version)
    if 'atom_features' in original and 'atom_features' in reconstructed:
        atom_feat_loss = 0
        for i in range(batch_size):
            orig_atoms = original['atom_features'][i]  # Size: [N_atoms, orig_atom_fea_len] 
            recon_atom_types = reconstructed['atom_types'][i]  # Size: [N_atoms, orig_atom_fea_len]        
            min_atoms = min(orig_atoms.size(0), recon_atom_types.size(0))        
            if min_atoms > 0:
                if orig_atoms.size(-1) == recon_atom_types.size(-1):
                    atom_feat_loss += F.mse_loss(
                        recon_atom_types[:min_atoms], 
                        orig_atoms[:min_atoms], 
                        reduction='sum'
                    )
                else:
                    print(f"Warning: Dimension mismatch - orig: {orig_atoms.size(-1)}, recon: {recon_atom_types.size(-1)}")
    
        reconstruction_loss += atom_feat_loss
        loss_components['atom_features_loss'] = atom_feat_loss.item()

    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'reconstruction_loss': reconstruction_loss,
        'kl_loss': kl_loss,
        'beta': beta,
        'components': loss_components
    }


class FlexibleVAETrainer:
    """
    Flexible trainer class for Crystal Graph VAE (with or without extra features)
    """
    
    def __init__(self, model, optimizer, device='cpu', beta_schedule=None, use_extra_features=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.beta_schedule = beta_schedule or (lambda epoch: 1.0)
        self.use_extra_features = use_extra_features
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_extra_feat_loss = 0
        
        beta = self.beta_schedule(epoch)
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle different batch formats (with or without extra features)
            if len(batch_data) == 3:  # (inputs, targets, cif_ids)
                inputs, targets, cif_ids = batch_data

                if len(inputs) == 5:  # Has extra features
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea = inputs
                    
                else:  # No extra features
                    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                    extra_fea = None
                    
            else:
                raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")


            
            # Move to device
            if self.device != 'cpu':
                atom_fea = atom_fea.to(self.device)
                nbr_fea = nbr_fea.to(self.device)
                nbr_fea_idx = nbr_fea_idx.to(self.device)
                crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]
                if extra_fea is not None:
                    extra_fea = extra_fea.to(self.device)
            
            # Forward pass
            output = self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea)
            
            # Prepare original data for loss calculation
            original = {
                'atom_features': [atom_fea[idx] for idx in crystal_atom_idx]
            }
            if extra_fea is not None:
                original['extra_features'] = extra_fea
            """
            # Calculate loss
            loss_dict = flexible_vae_loss_function(
                output['reconstructed'], 
                original, 
                output['mu'], 
                output['logvar'], 
                beta=beta,
                use_extra_features=self.use_extra_features
            )
            """
            if hasattr(self, 'balanced_training') and self.balanced_training:
                original_inputs = {}
                if extra_fea is not None:
                    original_inputs['extra_features'] = extra_fea
    
                loss_dict = balanced_vae_loss_function(
                    output, 
                    original_inputs, 
                    beta=beta,
                    structure_weight=getattr(self, 'structure_weight', 1.0),
                    feature_weight=getattr(self, 'feature_weight', 1.0))
            else:
                original = {'atom_features': [atom_fea[idx] for idx in crystal_atom_idx]}
                if extra_fea is not None:
                    original['extra_features'] = extra_fea    
                loss_dict = flexible_vae_loss_function(
                    output['reconstructed'], 
                    original, 
                    output['mu'], 
                    output['logvar'], 
                    beta=beta,
                    use_extra_features=self.use_extra_features)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            if hasattr(self, 'balanced_training') and self.balanced_training:
                total_recon_loss += (loss_dict.get('structure_loss', 0) + loss_dict.get('feature_loss', 0))
            if isinstance(total_recon_loss, torch.Tensor):
                total_recon_loss = total_recon_loss.item()
            total_extra_feat_loss += loss_dict.get('feature_loss', 0)
            if isinstance(total_extra_feat_loss, torch.Tensor):
                total_extra_feat_loss = total_extra_feat_loss.item()
            else:
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                if 'extra_features_loss' in loss_dict['components']:
                    total_extra_feat_loss += loss_dict['components']['extra_features_loss']
                total_kl_loss += loss_dict['kl_loss'].item()
 
           
            if batch_idx % 10 == 0:
                if hasattr(self, 'balanced_training') and self.balanced_training:
                    structure_loss = loss_dict.get('structure_loss', 0)
                    feature_loss = loss_dict.get('feature_loss', 0)
                    print(f'Batch {batch_idx}: Total={loss_dict["total_loss"].item():.1f}, '
                          f'Structure={structure_loss:.1f}, '
                          f'Features={feature_loss:.1f}, '
                          f'KL={loss_dict["kl_loss"].item():.1f}, Beta={beta:.4f}')
                else:
                    extra_info = ""
                    if self.use_extra_features and 'extra_features_loss' in loss_dict['components']:
                        extra_info = f", ExtraFeat={loss_dict['components']['extra_features_loss']:.4f}"
                
                    print(f'Batch {batch_idx}: Loss={loss_dict["total_loss"].item():.4f}, '
                          f'Recon={loss_dict["reconstruction_loss"].item():.4f}, '
                          f'KL={loss_dict["kl_loss"].item():.4f}{extra_info}, Beta={beta:.4f}')
        
        return {
            'total_loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader),
            'kl_loss': total_kl_loss / len(dataloader),
            'extra_features_loss': total_extra_feat_loss / len(dataloader) if self.use_extra_features else 0}


# Utility functions for creating flexible models
def create_flexible_vae_model(dataset=None, orig_atom_fea_len=92, nbr_fea_len=41, 
                             n_extra_features=None, latent_dim=64, device='cpu',
                             atom_fea_len=128, n_conv=3, h_fea_len=256):
    """
    Create flexible VAE model that adapts to dataset properties
    """
    # Auto-detect properties if dataset is provided
    if dataset is not None:
        try:
            # Try to get dimensions from dataset
            structures, _, _ = dataset[0]
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            
            # Check if dataset has extra features
            if len(structures) >= 4:  # Has extra features
                n_extra_features = structures[3].shape[-1] if len(structures[3].shape) > 0 else len(dataset.feature_names)
                use_extra_features = True
                print(f"Dataset has extra features: {n_extra_features}")
            else:
                n_extra_features = 0
                use_extra_features = False
                print("Dataset has no extra features - structure-only mode")
                
        except Exception as e:
            print(f"Could not auto-detect dataset properties: {e}")
            print("Using provided parameters")
            use_extra_features = n_extra_features is not None and n_extra_features > 0
    else:
        use_extra_features = n_extra_features is not None and n_extra_features > 0
    
    model = FlexibleCrystalGraphVAE(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
        h_fea_len=h_fea_len,
        n_extra_features=n_extra_features,
        latent_dim=latent_dim,
        max_atoms=200,
        use_extra_features=use_extra_features
    )
    
    if device != 'cpu':
        model = model.to(device)
    
    return model


def beta_annealing_schedule(epoch, max_beta=1.0, annealing_epochs=50):
    """
    Beta annealing schedule for beta-VAE
    """
    if epoch < annealing_epochs:
        return max_beta * (epoch / annealing_epochs)
    else:
        return max_beta
