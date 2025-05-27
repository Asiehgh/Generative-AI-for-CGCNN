#!/usr/bin/env python3
"""
Balanced Crystal Graph VAE - Equal weighting of CGCNN structure and molecular features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import ConvLayer

class BalancedCrystalGraphVAE(nn.Module):
    """
    Balanced VAE that equally emphasizes:
    1. CGCNN structure reconstruction (atom features, bonds, crystal graphs)
    2. Molecular property reconstruction (extra features)
    """
    
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64, 
                 n_conv=3, h_fea_len=128, n_extra_features=None, 
                 latent_dim=64, max_atoms=200, use_extra_features=True,
                 structure_weight=1.0, feature_weight=1.0):
        super(BalancedCrystalGraphVAE, self).__init__()
        
        self.atom_fea_len = atom_fea_len
        self.latent_dim = latent_dim
        self.max_atoms = max_atoms
        self.n_extra_features = n_extra_features if n_extra_features is not None else 0
        self.use_extra_features = use_extra_features and (self.n_extra_features > 0)
        self.orig_atom_fea_len = orig_atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # IMPORTANT: Loss weighting for balanced learning
        self.structure_weight = structure_weight
        self.feature_weight = feature_weight
        
        print(f"Initializing Balanced CGCNN + Features VAE:")
        print(f"  Structure weight: {structure_weight}")
        print(f"  Feature weight: {feature_weight}")
        print(f"  Use extra features: {self.use_extra_features}")
        print(f"  Number of extra features: {self.n_extra_features}")
        print(f"  Latent dimension: {latent_dim}")
        
        # ============= ENCODER =============
        # CGCNN Structure Encoder
        self.atom_embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # Graph convolution layers (CGCNN core)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        
        # Structure feature processing
        self.structure_encoder = nn.Sequential(
            nn.Linear(atom_fea_len, h_fea_len // 2),
            nn.BatchNorm1d(h_fea_len // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Extra features encoder (if using)
        if self.use_extra_features:
            extra_hidden_dim = min(128, max(32, self.n_extra_features * 4))
            self.extra_feat_encoder = nn.Sequential(
                nn.Linear(self.n_extra_features, extra_hidden_dim),
                nn.BatchNorm1d(extra_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(extra_hidden_dim, h_fea_len // 2),
                nn.BatchNorm1d(h_fea_len // 2),
                nn.ReLU()
            )
            combined_input_dim = h_fea_len  # structure + features
        else:
            self.extra_feat_encoder = None
            combined_input_dim = h_fea_len // 2  # structure only
        
        # Combined encoding to latent space
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_input_dim, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Latent space mappings
        self.fc_mu = nn.Linear(h_fea_len, latent_dim)
        self.fc_logvar = nn.Linear(h_fea_len, latent_dim)
        
        # ============= DECODER =============
        # Latent to features
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, h_fea_len),
            nn.BatchNorm1d(h_fea_len),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # STRUCTURE DECODER (CGCNN reconstruction)
        self.structure_decoder = nn.Sequential(
            nn.Linear(h_fea_len // 2, atom_fea_len * 2),
            nn.ReLU(),
            nn.Linear(atom_fea_len * 2, atom_fea_len),
            nn.ReLU()
        )
        
        # Atom type reconstruction (crucial for CGCNN)
        self.atom_type_decoder = nn.Sequential(
            nn.Linear(atom_fea_len, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, orig_atom_fea_len),
            nn.Sigmoid()  # Probabilistic atom types
        )
        
        # Bond feature reconstruction
        self.bond_decoder = nn.Sequential(
            nn.Linear(atom_fea_len * 2, atom_fea_len),
            nn.ReLU(),
            nn.Linear(atom_fea_len, nbr_fea_len),
            nn.Sigmoid()
        )
        
        # Number of atoms predictor
        self.num_atoms_predictor = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.ReLU(),
            nn.Linear(h_fea_len // 2, 1),
            nn.Sigmoid()
        )
        
        # FEATURE DECODER (molecular properties)
        if self.use_extra_features:
            self.extra_feat_decoder = nn.Sequential(
                nn.Linear(h_fea_len // 2, extra_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(extra_hidden_dim, self.n_extra_features)
            )
        else:
            self.extra_feat_decoder = None
            
    def encode(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea=None):
        """
        Encode both crystal structure (via CGCNN) and molecular features
        """
        # CGCNN Structure Encoding
        atom_fea = self.atom_embedding(atom_fea)
        
        # Apply graph convolutions (core CGCNN)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pool to crystal-level features
        crystal_fea = self.pooling(atom_fea, crystal_atom_idx)
        structure_encoded = self.structure_encoder(crystal_fea)
        
        # Molecular Features Encoding (if available)
        if self.use_extra_features and extra_fea is not None:
            extra_encoded = self.extra_feat_encoder(extra_fea)
            # Combine structure + features
            combined_fea = torch.cat([structure_encoded, extra_encoded], dim=1)
        else:
            combined_fea = structure_encoded
        
        # Encode to latent space
        combined_fea = self.feature_combiner(combined_fea)
        mu = self.fc_mu(combined_fea)
        logvar = self.fc_logvar(combined_fea)
        
        return mu, logvar
    
    def decode(self, z, num_atoms=None):
        """
        Decode to both crystal structure and molecular features
        """
        batch_size = z.size(0)
        
        # Decode from latent space
        decoded_features = self.latent_decoder(z)
        
        # Split for different reconstruction tasks
        if self.use_extra_features:
            structure_part = decoded_features[:, :decoded_features.size(1)//2]
            feature_part = decoded_features[:, decoded_features.size(1)//2:]
            
            # Reconstruct molecular features
            reconstructed_extra_fea = self.extra_feat_decoder(feature_part)
        else:
            structure_part = decoded_features
            reconstructed_extra_fea = None
        
        # Reconstruct crystal structure
        # Predict number of atoms
        if num_atoms is None:
            num_atoms_raw = self.num_atoms_predictor(decoded_features)
            num_atoms = (num_atoms_raw * self.max_atoms).round().long()
            num_atoms = torch.clamp(num_atoms, min=1, max=self.max_atoms)
        
        max_atoms_in_batch = num_atoms.max().item()
        
        # Generate atom features
        expanded_structure = structure_part.unsqueeze(1).expand(
            batch_size, max_atoms_in_batch, -1
        )
        
        # Add positional encoding for atoms
        pos_encoding = torch.arange(max_atoms_in_batch, device=z.device).float()
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)
        pos_encoding = pos_encoding / max_atoms_in_batch
        
        # Decode atom features
        atom_features = self.structure_decoder(
            expanded_structure.view(-1, expanded_structure.size(2))
        )
        atom_features = atom_features.view(batch_size, max_atoms_in_batch, -1)
        
        # Decode atom types (crucial for CGCNN)
        atom_types = self.atom_type_decoder(
            atom_features.view(-1, atom_features.size(2))
        )
        atom_types = atom_types.view(batch_size, max_atoms_in_batch, -1)
        
        # Generate bond features
        bond_features = self._generate_bond_features(atom_features, num_atoms)
        
        result = {
            'atom_features': atom_features,
            'atom_types': atom_types,
            'num_atoms': num_atoms,
            'bond_features': bond_features
        }
        
        if self.use_extra_features:
            result['extra_features'] = reconstructed_extra_fea
        
        return result
    
    def _generate_bond_features(self, atom_features, num_atoms):
        """Generate bond features for crystal structure"""
        batch_size = atom_features.size(0)
        bond_features_list = []
        
        for i in range(batch_size):
            n_atoms = num_atoms[i].item()
            if n_atoms > 1:
                bonds = []
                for j in range(min(n_atoms, 20)):  # Limit for efficiency
                    for k in range(min(n_atoms, 20)):
                        if j != k:
                            bond_input = torch.cat([
                                atom_features[i, j], 
                                atom_features[i, k]
                            ])
                            bonds.append(bond_input)
                
                if bonds:
                    bonds = torch.stack(bonds)
                    bond_feats = self.bond_decoder(bonds)
                    bond_features_list.append(bond_feats)
                else:
                    bond_features_list.append(
                        torch.zeros(1, self.nbr_fea_len, device=atom_features.device)
                    )
            else:
                bond_features_list.append(
                    torch.zeros(1, self.nbr_fea_len, device=atom_features.device)
                )
        
        return bond_features_list
    
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, extra_fea=None):
        """Forward pass with balanced structure + feature learning"""
        # Ensure float32
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
            'z': z,
            'original_atom_fea': atom_fea,
            'original_crystal_idx': crystal_atom_idx
        }
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """Crystal-level pooling from atom features"""
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
    
    def generate(self, num_samples=1, device='cpu'):
        """Generate new structures with both crystal and molecular properties"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            generated = self.decode(z)
        return generated


def balanced_vae_loss_function(output, original_inputs, beta=1.0, 
                             structure_weight=1.0, feature_weight=1.0):
    """
    Balanced loss function that equally weights structure and feature reconstruction
    """
    reconstructed = output['reconstructed']
    mu = output['mu']
    logvar = output['logvar']
    
    batch_size = mu.size(0)
    total_loss = 0
    loss_components = {}
    
    # ========== STRUCTURE RECONSTRUCTION LOSSES ==========
    structure_loss = 0
    
    # 1. Atom Features Loss (CGCNN structure learning)
    if 'original_atom_fea' in output:
        original_atom_fea = output['original_atom_fea']
        crystal_atom_idx = output['original_crystal_idx']
        
        atom_recon_loss = 0
        for i, idx_map in enumerate(crystal_atom_idx):
            if i < reconstructed['atom_types'].size(0):
                # Compare original vs reconstructed atom features
                orig_atoms = original_atom_fea[idx_map]
                recon_atoms = reconstructed['atom_types'][i][:len(idx_map)]
                
                if orig_atoms.size(-1) == recon_atoms.size(-1):
                    atom_recon_loss += F.mse_loss(recon_atoms, orig_atoms, reduction='sum')
        
        structure_loss += atom_recon_loss
        loss_components['atom_reconstruction'] = atom_recon_loss.item()
    
    # 2. Crystal Structure Consistency Loss
    if 'num_atoms' in reconstructed:
        # Penalty for unrealistic atom counts
        num_atoms = reconstructed['num_atoms'].float()
        atom_count_loss = F.mse_loss(num_atoms, torch.clamp(num_atoms, 10, 50), reduction='sum')
        structure_loss += atom_count_loss * 0.1
        loss_components['atom_count_consistency'] = atom_count_loss.item()
    
    # ========== FEATURE RECONSTRUCTION LOSS ==========
    feature_loss = 0
    if 'extra_features' in reconstructed and 'extra_features' in original_inputs:
        feature_loss = F.mse_loss(
            reconstructed['extra_features'], 
            original_inputs['extra_features'], 
            reduction='sum'
        )
        loss_components['feature_reconstruction'] = feature_loss.item()
    
    # ========== KL DIVERGENCE LOSS ==========
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # ========== BALANCED TOTAL LOSS ==========
    weighted_structure_loss = structure_weight * structure_loss
    weighted_feature_loss = feature_weight * feature_loss
    total_reconstruction_loss = weighted_structure_loss + weighted_feature_loss
    
    total_loss = total_reconstruction_loss + beta * kl_loss
    
    return {
        'total_loss': total_loss,
        'structure_loss': structure_loss,
        'feature_loss': feature_loss,
        'kl_loss': kl_loss,
        'beta': beta,
        'structure_weight': structure_weight,
        'feature_weight': feature_weight,
        'components': loss_components
    }
