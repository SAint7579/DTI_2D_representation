import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import math
import random
##### LOSS FUNCTIONS #####

### BASIC VAE LOSS ### 
def beta_vae_loss(x, x_recon, mu, logvar, beta=1.0, recon_type="mse", reduction="sum"):
    """
    Computes the β-VAE loss as the sum of a reconstruction loss and a weighted KL divergence.

    Args:
        x (Tensor): Original input [B, C, H, W].
        x_recon (Tensor): Reconstructed output [B, C, H, W].
        mu (Tensor): Mean of q(z|x) [B, latent_dim].
        logvar (Tensor): Log-variance of q(z|x) [B, latent_dim].
        beta (float): Weight for the KL divergence term.
        recon_type (str): "mse" or "bce" for reconstruction loss.
        reduction (str): "sum" or "mean" for final loss.

    Returns:
        total_loss (Tensor): Reconstruction loss + β * KL divergence.
        recon_loss (Tensor): Reconstruction loss.
        kl_loss (Tensor): Weighted KL divergence term (β * KL).
    """
    # 1. Reconstruction Loss
    if recon_type == "bce":
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction=reduction)
    else:
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)

    # 2. KL Divergence for a diagonal Gaussian
    # Compute KL divergence per sample: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)

    # Apply reduction over the batch
    if reduction == "sum":
        kl_loss = kl_loss.sum()
    elif reduction == "mean":
        kl_loss = kl_loss.mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, beta * kl_loss


def beta_tcvae_loss(x, x_recon, z, mu, logvar, beta=10.0, recon_type="mse", reduction="sum"):
    """
    Computes the β-TCVAE loss, which decomposes the KL divergence into:
        - Mutual Information (MI)
        - Total Correlation (TC, weighted by β)
        - Dimension-wise KL divergence

    Args:
        x (Tensor): Original input [B, C, H, W].
        x_recon (Tensor): Reconstructed output [B, C, H, W].
        z (Tensor): Sampled latent variables from q(z|x) [B, latent_dim].
        mu (Tensor): Mean of q(z|x) [B, latent_dim].
        logvar (Tensor): Log-variance of q(z|x) [B, latent_dim].
        beta (float): Weight for the TC term (default: 5.0).
        recon_type (str): "mse" or "bce" for reconstruction loss.
        reduction (str): "sum" or "mean" for final loss.

    Returns:
        total_loss (Tensor): recon_loss + weighted KL components
        recon_loss (Tensor): Reconstruction loss
        kl_loss (Tensor): MI + β*TC + dimension-wise KL
    """
    batch_size, latent_dim = mu.shape

    # 1. Reconstruction Loss (same as VAE)
    if recon_type == "bce":
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction=reduction)
    else:
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)

    # 2. KL Decomposition into MI, TC, and Dimension-wise KL
    # Precompute terms for log probabilities
    var = torch.exp(logvar)
    z_expanded = z.unsqueeze(1)  # [B, 1, D]
    mu_expanded = mu.unsqueeze(0)  # [1, B, D]
    logvar_expanded = logvar.unsqueeze(0)  # [1, B, D]

    # Compute log(q(z|x_j)) for all pairs (i, j) in the batch
    log_q_zx = -0.5 * (
        (z_expanded - mu_expanded).pow(2) / var.unsqueeze(0)
        + logvar_expanded
        + math.log(2 * math.pi)
    ).sum(dim=2)  # [B, B]

    # Mutual Information (MI): log(q(z|x)) - log(q(z))
    # Compute log(q(z)) ≈ log(1/(B-1) ∑_{j≠i} q(z_i|x_j))
    mask = torch.eye(batch_size, dtype=torch.bool, device=mu.device)
    log_q_z = torch.logsumexp(log_q_zx.masked_fill(mask, -float('inf')), dim=1) - math.log(batch_size - 1)
    mi_loss = (torch.diag(log_q_zx) - log_q_z)  # [B]

    # Total Correlation (TC): log(q(z)) - log(∏_j q(z_j))
    log_prod_q_z = 0.0
    for d in range(latent_dim):
        # Compute log(q(z_j)) for each dimension d
        log_q_zj = -0.5 * (
            (z[:, d].unsqueeze(1) - mu[:, d].unsqueeze(0)).pow(2) / var[:, d].unsqueeze(0)
            + logvar[:, d].unsqueeze(0)
            + math.log(2 * math.pi)
        )
        log_prod_q_z += torch.logsumexp(log_q_zj, dim=1) - math.log(batch_size)  # Sum over D
    tc_loss = log_q_z - log_prod_q_z  # [B]

    # Dimension-wise KL (KL(q(z) || p(z))
    dw_kl_loss = -0.5 * (1 + logvar - mu.pow(2) - var).sum(dim=1) - mi_loss - tc_loss  # [B]

    # Total KL loss = MI + β*TC + dimension-wise KL
    kl_loss = mi_loss + beta * tc_loss + dw_kl_loss

    # Apply reduction (sum or mean)
    if reduction == "sum":
        kl_loss = kl_loss.sum()
    elif reduction == "mean":
        kl_loss = kl_loss.mean()

    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss

def vae_loss(x, x_recon, mu, logvar, recon_type="mse", reduction="sum"):
    """
    Computes the standard VAE loss:
        1) Reconstruction loss (MSE or BCE)
        2) KL divergence
    Args:
        x (Tensor): Original input [B, C, H, W].
        x_recon (Tensor): Reconstructed output from the VAE [B, C, H, W].
        mu (Tensor): Mean of the approximate posterior [B, latent_dim].
        logvar (Tensor): Log-variance of the approximate posterior [B, latent_dim].
        recon_type (str): "mse" or "bce" for reconstruction term.
        reduction (str): "sum" or "mean" for the final loss. 
                         (Usually "sum" is used, then you divide by batch size afterwards.)
    Returns:
        total_loss (Tensor): Recon loss + KL divergence
        recon_loss (Tensor): The reconstruction component
        kl_loss (Tensor): The KL component
    """
    # --------------------------------------------------------
    # 1) Reconstruction Loss
    # --------------------------------------------------------
    if recon_type == "bce":
        # If x is in [0,1], we can use BCE
        # If x_recon was produced by a sigmoid, use that directly.
        # But if x_recon is raw logits, use 'binary_cross_entropy_with_logits'.
        # For simplicity, assume x_recon is already in [0,1].
        recon_loss = F.binary_cross_entropy(
            x_recon, x, reduction=reduction
        )
    else:
        # MSE reconstruction
        recon_loss = F.mse_loss(x_recon, x, reduction=reduction)

    # --------------------------------------------------------
    # 2) KL Divergence
    # --------------------------------------------------------
    # KL(N(mu, logvar), N(0, I)) = 
    #   -0.5 * \sum (1 + logvar - mu^2 - exp(logvar))
    # We'll do sum over batch dimension, then combine with reconstruction.
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # If you used reduction='mean', you'd need to decide how to average the KL as well.
    if reduction == "mean":
        kl_loss = kl_loss / x.size(0)

    # --------------------------------------------------------
    # 3) Total VAE Loss
    # --------------------------------------------------------
    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss

### SIMCLR LOSS ###
class RotationAugmentation90:
    def __init__(self):
        """
        Initializes the augmentation with fixed 90-degree rotation options.
        """
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        # If img is a tensor, convert it to a PIL image.
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        # Randomly choose one of the 90 degree multiples.
        angle = random.choice(self.angles)
        # Rotate the image by the selected angle (with expand=False to keep the same shape).
        rotated = TF.rotate(img, angle, expand=False)
        # Convert the rotated image back to a tensor.
        return transforms.ToTensor()(rotated)

# For a batch of images (if they are PIL.Images), you can process them like this:
def get_simclr_views_batch(images, augmentation=None):
    """
    Given a batch of images, generate two augmented views per image.
    
    Args:
        images (list[PIL.Image] or similar iterable): A batch of input images.
        augmentation (callable, optional): An augmentation pipeline. 
                                           If None, the default SimCLR pipeline is used.
    
    Returns:
        view1 (Tensor): A batch of first augmented views [B, C, H, W].
        view2 (Tensor): A batch of second augmented views [B, C, H, W].
    """

    ## Convert to PIL if the input is a list of tensors
    if isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(img) for img in images]

    if augmentation is None:
        augmentation = RotationAugmentation90()
    
    view1 = [augmentation(img) for img in images]
    view2 = [augmentation(img) for img in images]
    
    # Stack the list of tensors into a single tensor.
    view1 = torch.stack(view1)
    view2 = torch.stack(view2)
    return view1, view2

def simclr_loss(z_i, z_j, temperature=0.5, reduction="mean"):
    """
    Computes the SimCLR (InfoNCE) loss for a batch of paired embeddings.
    
    Args:
        z_i (Tensor): Embeddings from the first augmentation [B, D].
        z_j (Tensor): Embeddings from the second augmentation [B, D].
        temperature (float): Temperature scaling factor.
        reduction (str): Specifies the reduction to apply to the output: 
                         "mean" or "sum".
    
    Returns:
        loss (Tensor): The computed SimCLR loss.
    """
    batch_size = z_i.shape[0]
    
    # Normalize embeddings to unit norm.
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate embeddings: shape [2B, D]
    z = torch.cat([z_i, z_j], dim=0)
    
    # Compute cosine similarity matrix.
    sim_matrix = torch.matmul(z, z.T)
    logits = sim_matrix / temperature

    # Mask out self-similarities by setting them to a large negative value.
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    logits.masked_fill_(mask, -1e9)
    
    # For each example, the positive is the corresponding example in the other view.
    pos_idx = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (2 * batch_size)
    numerator = torch.exp(logits[torch.arange(2 * batch_size), pos_idx])
    denominator = torch.exp(logits).sum(dim=1)
    
    loss = -torch.log(numerator / denominator)
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError("Reduction must be either 'mean' or 'sum'.")
    
    return loss

def vae_simclr_loss(x1, x2, x1_recon, x2_recon, z1, z2, 
                    mu1, logvar1, mu2, logvar2, fm_feats1=None, fm_feats2=None,
                    recon_type="mse", reduction="sum",
                    simclr_temperature=0.5, 
                    vae_weight=1.0, simclr_weight=1.0, beta=10.0):
    """
    Computes a combined loss that adds together the VAE loss and the SimCLR loss.
    
    We assume that x1 and x2 are two augmented views of the same input image (or batch).
    Each view is passed through the VAE, resulting in reconstructions and latent parameters.
    The VAE loss is computed separately for each view and then averaged.
    For the SimCLR loss, we use the latent means (mu1 and mu2) as the representations.
    
    Args:
        x1 (Tensor): Original input view 1 [B, C, H, W].
        x2 (Tensor): Original input view 2 [B, C, H, W].
        x1_recon (Tensor): Reconstructed output from VAE for view 1 [B, C, H, W].
        x2_recon (Tensor): Reconstructed output from VAE for view 2 [B, C, H, W].
        mu1 (Tensor): Mean of the approximate posterior for view 1 [B, latent_dim].
        logvar1 (Tensor): Log-variance for view 1 [B, latent_dim].
        mu2 (Tensor): Mean of the approximate posterior for view 2 [B, latent_dim].
        logvar2 (Tensor): Log-variance for view 2 [B, latent_dim].
        recon_type (str): "mse" or "bce" for reconstruction loss.
        reduction (str): "sum" or "mean" for loss reduction.
        simclr_temperature (float): Temperature parameter for SimCLR loss.
        vae_weight (float): Weight for the VAE loss component.
        simclr_weight (float): Weight for the SimCLR loss component.
    
    Returns:
        total_loss (Tensor): The combined weighted loss.
        total_vae_loss (Tensor): Averaged VAE loss from both views.
        simclr_loss_val (Tensor): The SimCLR loss computed on the latent representations.
    """
    # Compute VAE loss for each view.
    loss1, recon_loss1, kl_loss1 = beta_tcvae_loss(x1, x1_recon, z1, mu1, logvar1, beta, recon_type, reduction)
    loss2, recon_loss2, kl_loss2 = beta_tcvae_loss(x2, x2_recon, z2, mu2, logvar2, beta, recon_type, reduction)
    
    # Average the VAE losses from both views.
    total_vae_loss = (loss1 + loss2) / 2.0

    total_recon_loss = (recon_loss1 + recon_loss2) / 2.0
    total_kl_loss = (kl_loss1 + kl_loss2) / 2.0
    
    # Compute SimCLR loss using the latent means as representations.
    if fm_feats1 is not None and fm_feats2 is not None:
        mu1 = torch.cat([mu1, fm_feats1], dim=1)
        mu2 = torch.cat([mu2, fm_feats2], dim=1)
    simclr_loss_val = simclr_loss(mu1, mu2, temperature=simclr_temperature, reduction=reduction)
    
    # Combine the two losses with their respective weights.
    total_loss = vae_weight * total_vae_loss + simclr_weight * simclr_loss_val
    
    return total_loss, total_vae_loss, total_recon_loss, total_kl_loss, simclr_loss_val


### BATCH HARD TRIPLET LOSS ###

def batch_hard_triplet_loss(embeddings, labels, margin=1.0, reduction='mean'):
    """
    Computes the batch hard triplet loss.
    For each anchor, the hardest positive (furthest in feature space among those sharing the same label)
    and the hardest negative (closest among those with a different label) are chosen.
    
    Args:
        embeddings (Tensor): Embeddings of shape [B, D].
        labels (Tensor): Labels of shape [B]. Expected to be integer type.
        margin (float): Margin for triplet loss.
        reduction (str): 'mean' or 'sum' over the batch.
        
    Returns:
        loss (Tensor): Scalar loss value.
    """
    labels = labels.long()  # ensure labels are integer type
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

    # Build positive mask: same labels, excluding self-comparisons.
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    diag = torch.eye(positive_mask.size(0), dtype=torch.bool, device=embeddings.device)
    positive_mask = positive_mask & ~diag

    # Build negative mask: different labels.
    negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1))

    # For each anchor, select the hardest positive (largest distance).
    # If no positive exists for an anchor, the distance defaults to zero.
    if positive_mask.sum() > 0:
        hardest_positive_dist, _ = (pairwise_dist * positive_mask.float()).max(dim=1)
    else:
        hardest_positive_dist = torch.zeros(embeddings.size(0), device=embeddings.device)

    # For negatives, set non-valid entries to a large value so they are ignored in min().
    max_dist = pairwise_dist.max().item()
    masked_negatives = pairwise_dist.clone()
    masked_negatives[~negative_mask] = max_dist + 1.0
    hardest_negative_dist, _ = masked_negatives.min(dim=1)

    losses = F.relu(hardest_positive_dist - hardest_negative_dist + margin)

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        return losses
    

def autoencoder_triplet_loss(x, x_recon, y, z, mu, logvar, fm_feats=None, margin=1.0, recon_type="mse", reduction="sum", triplet_weight=1.0, vae_weight=1.0, beta=10.0):
    """
    Combines the VAE loss with the triplet loss computed on the latent means.
    
    Args:
        x (Tensor): Original input images.
        x_recon (Tensor): Reconstructed images from the autoencoder.
        y (Tensor): Ground-truth labels.
        mu (Tensor): Latent means.
        logvar (Tensor): Latent log variances.
        margin (float): Margin for triplet loss.
        recon_type (str): "mse" or "bce" for the reconstruction loss.
        reduction (str): "sum" or "mean" for loss reduction.
        
    Returns:
        total_loss, total_vae_loss, recon_loss, kl_loss, triplet_loss_val
    """
    total_vae_loss, recon_loss, kl_loss = beta_tcvae_loss(x, x_recon, z, mu, logvar, beta, recon_type, reduction)

    if fm_feats is not None:
        mu = torch.cat([mu, fm_feats], dim=1)

    triplet_loss_val = batch_hard_triplet_loss(mu, y, margin=margin, reduction=reduction)
    total_loss = vae_weight * total_vae_loss + triplet_weight * triplet_loss_val
    return total_loss, total_vae_loss, recon_loss, kl_loss, triplet_loss_val

### AUTOENCODER + CLASSIFIER LOSS ###

def autoencoder_loss(x, x_recon, y, y_pred, z, mu, logvar, l1_matrix = None, class_loss_weight=1.0, vae_loss_weight=1.0, beta=10.0, l1_weight=0.01):
    '''
    Combined (weighted loss) for autoencoder and classifier

    Args:
        x (Tensor): Original input [B, C, H, W].
        y (Tensor): True labels [B].
        x_recon (Tensor): Reconstructed output from the VAE [B, C, H, W].
        mu (Tensor): Mean of the approximate posterior [B, latent_dim].
        logvar (Tensor): Log-variance of the approximate posterior [B, latent_dim].

    Returns:
        total_loss (Tensor): Recon loss + KL divergence + classifier loss
        recon_loss (Tensor): The reconstruction component
        kl_loss (Tensor): The KL component
        class_loss (Tensor): The classifier component  
    '''

    # VAE loss
    total_vae_loss, recon_loss, kl_loss = beta_tcvae_loss(x, x_recon, z, mu, logvar, recon_type="mse", reduction="sum", beta=beta)
    # total_vae_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, recon_type="mse", reduction="sum")

    # Classifier loss
    criterion = nn.BCEWithLogitsLoss()
    class_loss = criterion(y_pred, y)

    # L1 regularization on latent vector
    if l1_matrix is not None:
        l1_loss = torch.norm(l1_matrix, p=1, dim=1).mean()
    else:
        l1_loss = torch.tensor(0.0)

    # Combine the losses
    total_loss = (vae_loss_weight * total_vae_loss + 
                class_loss_weight * class_loss + 
                l1_weight * l1_loss)

    return total_loss, total_vae_loss, recon_loss, kl_loss, class_loss, l1_loss



### MODEL DEFINITION ###
class DTI_autoencoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 latent_dim=32, 
                 num_classes=10):
        """
        Variational Autoencoder with a classifier head.
        Encoder: 9x9 -> 1x1 -> produces mu, logvar
        Reparam trick: z = mu + eps * exp(0.5 * logvar)
        Decoder: z -> recon(9x9)
        Classifier: z -> class logits
        """
        super(DTI_autoencoder, self).__init__()

        # -------------------------
        #        Encoder
        # -------------------------
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 9 -> 4

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4 -> 2

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 2 -> 1

        # Flatten (64, 1, 1) -> 64
        self.fc_enc = nn.Linear(64, 128)

        # For VAE, we have two separate linear layers for mu and logvar
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # -------------------------
        #      Decoder
        # -------------------------
        self.fc_dec = nn.Linear(latent_dim, 64)  # map z back to shape (64,1,1)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 1 -> 2
        self.relu4   = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 2 -> 4
        self.relu5   = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(16, in_channels, kernel_size=4,
                                          stride=2, padding=1, output_padding=1)  # 4 -> 9

        # -------------------------
        #     Classifier Head
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  # final logits
        )

    def encode(self, x):
        """Encode input x -> (mu, logvar)."""
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # flatten
        x = x.view(x.size(0), -1)  # shape: (batch_size, 64)

        x = self.fc_enc(x)         # shape: (batch_size, 128)
        mu = self.fc_mu(x)         # shape: (batch_size, latent_dim)
        logvar = self.fc_logvar(x) # shape: (batch_size, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + eps * exp(0.5 * logvar)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        """Decode z -> reconstructed x."""
        x = self.fc_dec(z)            # shape: (batch_size, 64)
        x = x.view(x.size(0), 64, 1, 1)

        x = self.deconv1(x)           # shape: (32, 2, 2)
        x = self.relu4(x)

        x = self.deconv2(x)           # shape: (16, 4, 4)
        x = self.relu5(x)

        x = self.deconv3(x)           # shape: (in_channels, 9, 9)
        # Optionally apply sigmoid if your input is scaled 0-1
        # x = torch.sigmoid(x)
        return x

    def forward(self, x):
        """
        Forward pass:
        1) Encode -> mu, logvar
        2) Reparameterize -> z
        3) Decode -> x_recon
        4) Classify -> class_logits

        Returns:
            x_recon (Tensor): Reconstructed input
            class_logits (Tensor): Class logits from classifier
            mu (Tensor): Mean of latent distribution
            logvar (Tensor): Log-variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        class_logits = self.classifier(mu)

        return x_recon, class_logits, mu, logvar

class FMLayer2ndOrder(nn.Module):
    def __init__(self, num_features: int, latent_dim: int):
        super().__init__()
        # no bias or linear term here; pure 2-way interactions
        self.V = nn.Parameter(torch.randn(num_features, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, num_features]
        # sum-square trick for 2nd-order interactions
        xv = x @ self.V                       # [B, k]
        xv_sq = xv * xv                       # [B, k]
        x_sq = x * x                          # [B, n]
        v_sq = self.V * self.V                # [n, k]
        x_v_sq = x_sq @ v_sq                  # [B, k]
        interaction = xv_sq - x_v_sq          # [B, k]
        return interaction                    # [B, k]

class Spatial_DTI_autoencoder(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 latent_dim=16, 
                 num_classes=10,
                 fm_latent_dim=None):
        super().__init__()
        # -------------------------
        #      Encoder
        # -------------------------
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 9→9
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # 9→4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # 4→4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                                        # 4→2
        )
        # μ and logvar convs (1×1), output shape: (B, latent_dim, 2, 2)
        self.conv_mu     = nn.Conv2d(64, latent_dim,     kernel_size=1)
        self.conv_logvar = nn.Conv2d(64, latent_dim,     kernel_size=1)

        # -------------------------
        #      Decoder (Spatial Broadcast)
        # -------------------------
        self.dec_conv = nn.Sequential(
            nn.Conv2d(latent_dim + 2, 128, kernel_size=3, padding=1),  # with XY channels
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)       # → (B,1,9,9)
        )
        
        # -------------------------
        #     Factorization Machine
        # -------------------------
        if fm_latent_dim is not None:
            self.fm_latent_dim = fm_latent_dim
            self.fm = FMLayer2ndOrder(num_features=81, latent_dim=self.fm_latent_dim)

        else:
            self.fm = None
            self.fm_latent_dim = None

        # -------------------------
        #     Classifier Head
        # -------------------------
        if fm_latent_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim * 2 * 2 + fm_latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim * 2 * 2, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_classes)
            )

    def encode(self, x, with_fm=False):
        """x: (B,1,9,9) → mu, logvar: (B,latent,2,2)"""
        h = self.enc_conv(x)
        mu     = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        
        ## Concatenate fm with mu
        if with_fm:
            B = x.size(0)
            flat = x.view(B, -1)                    # [B,81]
            fm_feats = self.fm(flat)                # [B, k]
            new_mu = torch.cat([mu.view(B, -1), fm_feats], dim=1)
            return mu, logvar, new_mu
        else:
            return mu, logvar

    def reparameterize(self, mu, logvar):
        """Perform sampling per spatial location."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def spatial_broadcast(self, z):
        """
        Upsample z: (B,C,2,2) → (B,C,9,9), 
        build XY channels, and concat → (B,C+2,9,9).
        """
        B, C, _, _ = z.size()
        # Upsample latent to full image size
        z_upsampled = F.interpolate(z, size=(9,9), mode='nearest')  # :contentReference[oaicite:3]{index=3}

        # Create normalized coordinate maps
        xs = torch.linspace(-1, 1, 9, device=z.device)
        ys = torch.linspace(-1, 1, 9, device=z.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')     # shape (9,9)
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).unsqueeze(1) # (B,1,9,9)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).unsqueeze(1) # (B,1,9,9)

        # Concatenate along channel dim
        return torch.cat([z_upsampled, grid_x, grid_y], dim=1)

    def decode(self, z):
        """
        z: (B,latent,2,2) → recon: (B,1,9,9)
        """
        sb = self.spatial_broadcast(z)
        return self.dec_conv(sb)  # :contentReference[oaicite:4]{index=4}

    def classify(self, mu, fm_feats=None):
        """
        mu: (B,latent,2,2) → logits: (B,num_classes)
        """
        # Global average pool over spatial dims → (B,latent)
        # pooled = mu.mean(dim=[2,3])
        flattened = mu.view(mu.size(0), -1)
        if fm_feats is not None:
            flattened = torch.cat([flattened, fm_feats], dim=1)
        return self.classifier(flattened)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)

        ## Concatenate FM features with mu
        if self.fm_latent_dim is not None:
            # Flatten pixels for FM
            B = x.size(0)
            flat = x.view(B, -1)                    # [B,81]
            fm_feats = self.fm(flat)                # [B, k]
        else:
            fm_feats = None

        logits = self.classify(mu, fm_feats)

        ## Flatten 2x2xlatent to latent
        mu = mu.view(mu.size(0), -1)
        logvar = logvar.view(logvar.size(0), -1)
        if self.fm_latent_dim is not None:
            return recon, logits, mu, logvar, fm_feats
        else:
            return recon, logits, mu, logvar
        
    def get_interactions(self, x):
        B = x.size(0)
        flat = x.view(B, -1)                    # [B,81]
        fm_feats = self.fm(flat)                # [B, k]        # Assemble full 81x81 grid if requested
        G = self.fm.V @ self.fm.V.t()         # [81,81]
        pixel_outer = flat.unsqueeze(2) * flat.unsqueeze(1)
        interactions = pixel_outer * G.unsqueeze(0)  # [B,81,81]
        return interactions
    
if __name__ == '__main__':
    # Test the model
    model = DTI_autoencoder(in_channels=1, latent_dim=32, num_classes=1)
    x = torch.randn(16, 1, 9, 9)
    x_recon, class_logits, mu, logvar = model(x)

    print("Input shape:", x.shape)
    print("Reconstructed shape:", x_recon.shape)
    print("Class logits shape:", class_logits.shape)
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)