import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
from torchvision.models.video import r3d_18

##### LOSS FUNCTIONS #####

### BASIC VAE LOSS ### 

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

def vae_simclr_loss(x1, x2, x1_recon, x2_recon, 
                    mu1, logvar1, mu2, logvar2,
                    recon_type="mse", reduction="sum",
                    simclr_temperature=0.5, 
                    vae_weight=1.0, simclr_weight=1.0):
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
    loss1, recon_loss1, kl_loss1 = vae_loss(x1, x1_recon, mu1, logvar1, recon_type, reduction)
    loss2, recon_loss2, kl_loss2 = vae_loss(x2, x2_recon, mu2, logvar2, recon_type, reduction)
    
    # Average the VAE losses from both views.
    total_vae_loss = (loss1 + loss2) / 2.0

    total_recon_loss = (recon_loss1 + recon_loss2) / 2.0
    total_kl_loss = (kl_loss1 + kl_loss2) / 2.0
    
    # Compute SimCLR loss using the latent means as representations.
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
    

def autoencoder_triplet_loss(x, x_recon, y, mu, logvar, margin=1.0, recon_type="mse", reduction="sum", triplet_weight=1.0, vae_weight=1.0):
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
    total_vae_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, recon_type, reduction)
    triplet_loss_val = batch_hard_triplet_loss(mu, y, margin=margin, reduction=reduction)
    total_loss = vae_weight * total_vae_loss + triplet_weight * triplet_loss_val
    return total_loss, total_vae_loss, recon_loss, kl_loss, triplet_loss_val

### AUTOENCODER + CLASSIFIER LOSS ###

def autoencoder_loss(x, x_recon, y, y_pred, mu, logvar, class_loss_weight=1.0, vae_loss_weight=1.0):
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
    total_vae_loss, recon_loss, kl_loss = vae_loss(x, x_recon, mu, logvar, recon_type="mse", reduction="sum")

    # Classifier loss
    criterion = nn.BCEWithLogitsLoss()
    class_loss = criterion(y_pred, y)

    # Combine the losses
    total_loss = vae_loss_weight * total_vae_loss + class_loss_weight * class_loss

    return total_loss, total_vae_loss, recon_loss, kl_loss, class_loss



### MODEL DEFINITION ###
class MRE_autoencoder(nn.Module):
    def __init__(self, 
                 latent_dim=32,
                 num_classes=1,
                 input_shape=(96, 96, 48),
                 pretrained=True):
        super(MRE_autoencoder, self).__init__()
        
        # -------------------------
        #        Encoder
        # -------------------------
        # Load pretrained ResNet3D
        self.resnet = r3d_18(pretrained=pretrained)
        
        # Freeze all parameters except those we modify
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Modify first conv layer for single channel input
        self.resnet.stem[0] = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove original FC layer and add VAE heads
        self.resnet.fc = nn.Identity()  # We'll handle the encoding manually
        
        # Calculate final feature volume
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            features = self.resnet(dummy_input)
            self.feature_dim = features.shape[1]
            self.feature_shape = features.shape[2:]
            
        # VAE projection heads
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        
        # -------------------------
        #        Decoder
        # -------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Unflatten(1, (512, 1, 1, 1)),
            
            nn.ConvTranspose3d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(size=input_shape, mode='trilinear', align_corners=False)
        )
        
        # -------------------------
        #     Classifier Head
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def encode(self, x):
        # Forward through ResNet backbone
        features = self.resnet(x)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        # Calculate latent parameters
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        class_logits = self.classifier(z)
        return x_recon, class_logits, mu, logvar

if __name__ == '__main__':
    model = MRE_autoencoder(latent_dim=32, num_classes=1, input_shape=(96, 96, 48))
    model.eval()  # Set to evaluation mode

    # Create a dummy input tensor (batch_size=1, channels=1, depth=96, height=96, width=48)
    dummy_input = torch.randn(1, 1, 96, 96, 48)

    # Forward pass
    with torch.no_grad():
        x_recon, class_logits, mu, logvar = model(dummy_input)

    # Print output shapes
    print("Input shape:", dummy_input.shape)
    print("\nEncoder outputs:")
    print("Mu shape:", mu.shape)
    print("Logvar shape:", logvar.shape)
    print("\nDecoder output:")
    print("Reconstruction shape:", x_recon.shape)
    print("\nClassifier output:")
    print("Class logits shape:", class_logits.shape)

    # Additional sanity checks
    print("\nSanity checks:")
    print("Mu and logvar same shape?", mu.shape == logvar.shape)
    print("Reconstruction matches input shape?", x_recon.shape == dummy_input.shape)
    print("Latent dimension:", mu.shape[1])