import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Multimodal_backbone(nn.Module):
    '''
    Backbone combining multiple modalities
    '''

    def __init__(self, modality_encoder_tupple, clinical_params_size, latent_dim_sizes):
        super().__init__()
        self.modality_encoder_tupple = modality_encoder_tupple

        ## Make the encoder models non-trainable
        for encoder_model in self.modality_encoder_tupple:
            for param in encoder_model.parameters():
                param.requires_grad = False

                

        ## Sequential DNN backbone for x_param
        # self.fc1 = nn.Linear(clinical_params_size, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 128)

        self.x_params_backbone = nn.Sequential(
            nn.Linear(clinical_params_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        ## Classifier head
        concat_len = 128 + sum(latent_dim_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(concat_len, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    
    def forward(self, x_modalities, x_params):

        ## Forward pass for each modality
        latent_tensors = []
        for i, modality in enumerate(x_modalities):
            encoder_model = self.modality_encoder_tupple[i]
            _, _, mu = encoder_model.encode(modality,with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            latent_tensors.append(mu)

        ## Forward pass for clinical params
        x_params = self.x_params_backbone(x_params)

        ## Concatenate the latent tensors and clinical params
        x = torch.cat([x_params] + latent_tensors, dim=1)

        ## Forward pass through the classifier
        x = self.classifier(x)

        return x
    

    def get_fused_latent(self, x_modalities, x_params):
        latent_tensors = []
        for i, modality in enumerate(x_modalities):
            encoder_model = self.modality_encoder_tupple[i]
            _, _, mu = encoder_model.encode(modality,with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            latent_tensors.append(mu)

        x_params = self.x_params_backbone(x_params)

        x = torch.cat([x_params] + latent_tensors, dim=1)

        return x
    

class MMTMBlock(nn.Module):
    """
    Multi-modal Transfer Module (MMTM) via squeeze-and-excitation.
    Provides interaction scores based on gradient sensitivities.
    """
    def __init__(self, input_dims, bottleneck_dim=None, reduction=4):
        super().__init__()
        total_dim = sum(input_dims)
        if bottleneck_dim is None:
            bottleneck_dim = max(1, total_dim // reduction)
        self.squeeze = nn.Linear(total_dim, bottleneck_dim)
        self.excitations = nn.ModuleList([
            nn.Linear(bottleneck_dim, dim)
            for dim in input_dims
        ])

    def forward(self, features):
        # Squeeze: concatenate all features
        U = torch.cat(features, dim=1)
        z = F.relu(self.squeeze(U))
        # Excitation (gating) + residual
        fused = []
        for feat, excite in zip(features, self.excitations):
            e = torch.sigmoid(excite(z))
            fused_feat = feat * e + feat
            fused.append(fused_feat)
        return fused

    def compute_gradient_interactions(self, features):
        """
        Compute gradient-based interaction scores between every output and input dimension.

        For fused outputs F and inputs X (flattened), calculates
            S_{ij} = average_n |d F_i[n] / d X_j[n]|,
        averaged over the batch.

        Returns:
            interactions (ndarray): [D_out, D_in] matrix of avg abs gradients.
        """
        # Ensure inputs require grad
        inputs = [f.clone().detach().requires_grad_(True) for f in features]

        # Perform forward to get fused outputs
        fused = self.forward(inputs)
        # Flatten fused outputs to [batch, D_out]
        flat_out = torch.cat([f.view(f.size(0), -1) for f in fused], dim=1)
        batch_size, D_out = flat_out.shape
        # Flatten inputs to [batch, D_in]
        flat_in_list = [f.view(f.size(0), -1) for f in inputs]
        flat_in = torch.cat(flat_in_list, dim=1)
        _, D_in = flat_in.shape

        # Initialize interactions
        interactions = torch.zeros(D_out, D_in, device=flat_out.device)

        # Compute per-example jacobians and accumulate
        for n in range(batch_size):
            # Define function for a single sample
            def single_forward(*in_samples):
                # in_samples: tuple of [dim_i] tensors
                feats = []
                for sample in in_samples:
                    feats.append(sample.unsqueeze(0))  # [1, dim_i]
                fused_sample = self.forward(feats)
                # Flatten fused outputs: list of [1, dim]
                return torch.cat([f.squeeze(0) for f in fused_sample], dim=0)  # [D_out]

            # Prepare per-sample inputs: tuple of [dim_i]
            in_n = tuple(inp[n] for inp in inputs)
            # Compute jacobian: returns tuple of [D_out, dim_i] for each input
            Jn = torch.autograd.functional.jacobian(single_forward, in_n)
            # Concatenate gradients to [D_out, D_in]
            Jn_flat = torch.cat([j for j in Jn], dim=1)
            # Accumulate absolute gradients
            interactions += Jn_flat.abs()

        # Average over batch
        interactions = interactions / batch_size
        return interactions.detach().cpu().numpy()
    
    def permutation_threshold(self, features, n_permutations=200,
    alpha=0.05,
    shuffle_dim=0):
        """
        Compute permutation-based threshold for feature importance.
        """
        # Get original interaction matrix
        interactions = self.compute_gradient_interactions(features)
        # 1) compute the real interactions
        real_S = self.compute_gradient_interactions(features)  # numpy (D_out, D_in)

        D_out, D_in = real_S.shape
        thresholds = np.zeros_like(real_S)

        # 2) for each input channel j, build a null by shuffling that channel
        for j in range(D_in):
            null_vals = np.zeros((n_permutations, D_out), dtype=float)

            for p in range(n_permutations):
                # 2a) deep‐copy and shuffle the j-th flattened input across the batch
                perm_features = []
                for idx, f in enumerate(features):
                    # flatten each feature to (B, dim) for shuffling
                    B = f.shape[0]
                    flat = f.reshape(B, -1).clone()

                    if idx == j:
                        perm_idx = torch.randperm(B)
                        flat = flat[perm_idx]

                    # restore original shape
                    perm_features.append(flat.reshape_as(f))

                # 2b) compute interactions under this permutation
                S_perm = self.compute_gradient_interactions(perm_features)
                null_vals[p] = S_perm[:, j]   # collect only column j

            # 2c) threshold per output‐channel i
            # e.g. 95th percentile of null_vals[:, i]
            pct = 100 * (1 - alpha)
            thresholds[:, j] = np.percentile(null_vals, pct, axis=0)

        # 3) significance mask
        significant = real_S > thresholds

        return real_S, thresholds, significant
        

class MultimodalFusionBackbone(nn.Module):
    """
    Fuse modalities with MMTM and provide gradient-based interaction scores.
    """
    def __init__(
        self,
        modality_encoders: nn.ModuleList,
        clinical_params_size: int,
        latent_dim_sizes: list,
        clinical_emb_dim: int = 128,
        mmtm_reduction: int = 4,
    ):
        super().__init__()
        self.modality_encoders = modality_encoders
        # Freeze pretrained encoders
        for enc in self.modality_encoders:
            for p in enc.parameters():
                p.requires_grad = False

        # Clinical parameter transforms
        # self.param_pre = nn.Linear(clinical_params_size, clinical_emb_dim)
        self.param_post = nn.Sequential(
            nn.Linear(clinical_params_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )

        # MMTM fusion block
        all_dims = latent_dim_sizes + [clinical_params_size]
        self.mmtm = MMTMBlock(input_dims=all_dims, reduction=mmtm_reduction)

        # Classifier
        fused_dim = sum(latent_dim_sizes) + 128
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1)
        )

    def forward(self, x_modalities, x_params):
        # Encode and flatten modalities
        modality_latents = []
        for enc, mod in zip(self.modality_encoders, x_modalities):
            _, _, mu = enc.encode(mod, with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            modality_latents.append(mu)
        # Clinical pre-fuse
        # params_pre = self.param_pre(x_params)
        # Fuse all
        fused_list = self.mmtm(modality_latents + [x_params])
        fused_mods, fused_params = fused_list[:-1], fused_list[-1]
        # Clinical post-fuse
        params_post = self.param_post(fused_params)
        # Concat & classify
        x = torch.cat(fused_mods + [params_post], dim=1)
        return self.classifier(x)

    def get_gradient_interactions(self, x_modalities, x_params):
        """
        Compute [D_out x D_in] gradient interaction matrix for fused dims vs inputs.
        """
        # Encode and flatten modalities
        modality_latents = []
        for enc, mod in zip(self.modality_encoders, x_modalities):
            _, _, mu = enc.encode(mod, with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            modality_latents.append(mu)
        # Clinical pre-fuse
        # params_pre = self.param_pre(x_params)
        # Delegate to MMTMBlock
        return self.mmtm.compute_gradient_interactions(modality_latents + [x_params])
    
    def get_permutation_threshold(self, x_modalities, x_params, n_permutations=200, alpha=0.05, shuffle_dim=0):
        """
        Compute permutation-based threshold for feature importance.
        """
        # Encode and flatten modalities
        modality_latents = []
        for enc, mod in zip(self.modality_encoders, x_modalities):
            _, _, mu = enc.encode(mod, with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            modality_latents.append(mu)
        # Clinical pre-fuse
        # params_pre = self.param_pre(x_params)
        # Delegate to MMTMBlock
        return self.mmtm.permutation_threshold(modality_latents + [x_params], n_permutations, alpha, shuffle_dim)
    

    def aggregate_interactions(self, interactions, blocks, agg='mean'):
        """
        Aggregate full interaction matrix into modality-level [M x M].
        """
        import numpy as np
        boundaries = np.cumsum([0] + blocks)
        M = len(blocks)
        agg_mat = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                sub = interactions[boundaries[i]:boundaries[i+1], boundaries[j]:boundaries[j+1]]
                agg_mat[i, j] = sub.mean() if agg == 'mean' else sub.sum()
        return agg_mat

    def get_fused_latent(self, x_modalities, x_params):
        """
        Get the fused latent representation without classification.
        """
        # Encode modalities
        modality_latents = []
        for enc, mod in zip(self.modality_encoders, x_modalities):
            _, _, mu = enc.encode(mod, with_fm=True)
            if mu.dim() > 2:
                mu = mu.view(mu.size(0), -1)
            modality_latents.append(mu)
        # Clinical pre-fuse
        # params_pre = self.param_pre(x_params)
        # Fuse all
        fused_list = self.mmtm(modality_latents + [x_params])
        fused_mods, fused_params = fused_list[:-1], fused_list[-1]
        # Clinical post-fuse
        params_post = self.param_post(fused_params)
        # Concat & classify
        x = torch.cat(fused_mods + [params_post], dim=1)

        return x