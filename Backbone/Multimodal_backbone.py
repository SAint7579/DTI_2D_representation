import torch
import torch.nn as nn
import torch.nn.functional as F

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
            mu,_ = encoder_model.encode(modality)
            latent_tensors.append(mu)

        ## Forward pass for clinical params
        x_params = self.x_params_backbone(x_params)

        ## Concatenate the latent tensors and clinical params
        x = torch.cat([x_params] + latent_tensors, dim=1)

        ## Forward pass through the classifier
        x = self.classifier(x)

        return x