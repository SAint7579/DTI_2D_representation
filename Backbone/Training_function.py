import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute))

def train_autoencoder_aux(train_loader, test_loader, model, optimizer, n_epochs=100, batch_size=32, verbose=True):
    '''
    Function to train the axu classifier autoencoder model

    Parameters
    ----------
    X_train : np.array
        Training data
    y_train : np.array
        Training labels
    X_test : np.array
        Testing data
    y_test : np.array
        Testing labels
    model : torch.nn.Module
        Pytorch model
    optimizer : torch.optim
        Pytorch optimizer
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size

    Returns
    -------
    model : torch.nn.Module
        Trained model
    '''
    device = torch.device('cuda')
    model.to(device)
    best_loss = np.inf 
    es_counter = 0
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_vae_loss = 0
        train_loss_recon = 0
        train_loss_kl = 0
        train_loss_triplet = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            x_recon, class_logits, mu, logvar = model(X_batch)
            ## Flatten class logits
            class_logits = class_logits.view(-1)
            total_loss, total_vae_loss, recon_loss, kl_loss, t_loss = autoencoder_triplet_loss(
            X_batch, x_recon, y_batch, mu, logvar,
            margin=1.0, recon_type="mse", reduction="sum",
            triplet_weight=1.0, vae_weight=0.5
            )

            target_loss = total_loss
            target_loss.backward()
            optimizer.step()

            ## Getting the loss for training
            train_loss += total_loss.item()
            train_vae_loss += total_vae_loss.item()
            train_loss_recon += recon_loss.item()
            train_loss_kl += kl_loss.item()
            train_loss_triplet += t_loss.item()

        model.eval()
        test_loss = 0
        test_vae_loss = 0
        test_loss_recon = 0
        test_loss_kl = 0
        test_loss_triplet = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                x_recon, class_logits, mu, logvar = model(X_batch)
                class_logits = class_logits.view(-1)
                total_loss, total_vae_loss, recon_loss, kl_loss, t_loss = autoencoder_triplet_loss(
                    X_batch, x_recon, y_batch, mu, logvar,
                    margin=1.0, recon_type="mse", reduction="sum",
                    triplet_weight=1.0, vae_weight=0.5
                )
                ## Getting the loss for testing
                test_loss += total_loss.item()
                test_vae_loss += total_vae_loss.item()
                test_loss_recon += recon_loss.item()
                test_loss_kl += kl_loss.item()
                test_loss_triplet += t_loss.item()

        train_losses.append({'total_loss': train_loss/len(train_loader), 'vae_loss': train_vae_loss/len(train_loader), 'recon_loss': train_loss_recon/len(train_loader), 'kl_loss': train_loss_kl/len(train_loader), 'triplet_loss': train_loss_triplet/len(train_loader)})
        test_losses.append({'total_loss': test_loss/len(test_loader), 'vae_loss': test_vae_loss/len(train_loader), 'recon_loss': test_loss_recon/len(test_loader), 'kl_loss': test_loss_kl/len(test_loader), 'triplet_loss': test_loss_triplet/len(test_loader)})
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs}')
            print("Training")
            print(f'Total Loss: {train_loss/len(train_loader)}')
            print(f'Total VAE Loss: {train_vae_loss/len(train_loader)}')
            print(f'Recon Loss: {train_loss_recon/len(train_loader)}, KL Loss: {train_loss_kl/len(train_loader)}, Triplet Loss: {train_loss_triplet/len(train_loader)}')

            print("Testing")
            print(f'Total Loss: {test_loss/len(test_loader)}')
            print(f'Total VAE Loss: {test_vae_loss/len(train_loader)}')
            print(f'Recon Loss: {test_loss_recon/len(test_loader)}, KL Loss: {test_loss_kl/len(test_loader)}, Triplet Loss: {test_loss_triplet/len(test_loader)}')
            print('------------------------------------\n')

        ## Track the best model
        if test_loss < best_loss:
            best_loss = test_loss
            ## Save the model temporarily
            print("**** Saving the model ***\n")
            torch.save(model.state_dict(), '../Dataset/Saved_models/temp_best_model.pth')
            es_counter = 0
            
        else:
            es_counter += 1
            if es_counter == 100:
                print('Early stopping')
                print(f'Best Test loss: {best_loss}')
                ## Load the best model
                model.load_state_dict(torch.load('../Dataset/Saved_models/temp_best_model.pth'))
                return model, best_loss, {'train': train_losses, 'test': test_losses}
            
    return model, best_loss, {'train': train_losses, 'test': test_losses}


def train_autoencoder_triplet(train_loader, test_loader, model, optimizer, n_epochs=100, batch_size=32, verbose=True):
    '''
    Function to train a regression model

    Parameters
    ----------
    X_train : np.array
        Training data
    y_train : np.array
        Training labels
    X_test : np.array
        Testing data
    y_test : np.array
        Testing labels
    model : torch.nn.Module
        Pytorch model
    optimizer : torch.optim
        Pytorch optimizer
    n_epochs : int
        Number of epochs
    batch_size : int
        Batch size

    Returns
    -------
    model : torch.nn.Module
        Trained model
    '''
    device = torch.device('cuda')
    model.to(device)
    best_loss = np.inf 
    es_counter = 0
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_vae_loss = 0
        train_loss_recon = 0
        train_loss_kl = 0
        train_loss_triplet = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            x_recon, class_logits, mu, logvar = model(X_batch)
            ## Flatten class logits
            class_logits = class_logits.view(-1)
            total_loss, total_vae_loss, recon_loss, kl_loss, t_loss = autoencoder_triplet_loss(
            X_batch, x_recon, y_batch, mu, logvar,
            margin=1.0, recon_type="mse", reduction="sum",
            triplet_weight=1.0, vae_weight=0.5
            )

            target_loss = total_loss
            target_loss.backward()
            optimizer.step()

            ## Getting the loss for training
            train_loss += total_loss.item()
            train_vae_loss += total_vae_loss.item()
            train_loss_recon += recon_loss.item()
            train_loss_kl += kl_loss.item()
            train_loss_triplet += t_loss.item()

        model.eval()
        test_loss = 0
        test_vae_loss = 0
        test_loss_recon = 0
        test_loss_kl = 0
        test_loss_triplet = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                x_recon, class_logits, mu, logvar = model(X_batch)
                class_logits = class_logits.view(-1)
                total_loss, total_vae_loss, recon_loss, kl_loss, t_loss = autoencoder_triplet_loss(
                    X_batch, x_recon, y_batch, mu, logvar,
                    margin=1.0, recon_type="mse", reduction="sum",
                    triplet_weight=1.0, vae_weight=0.5
                )
                ## Getting the loss for testing
                test_loss += total_loss.item()
                test_vae_loss += total_vae_loss.item()
                test_loss_recon += recon_loss.item()
                test_loss_kl += kl_loss.item()
                test_loss_triplet += t_loss.item()

        train_losses.append({'total_loss': train_loss/len(train_loader), 'vae_loss': train_vae_loss/len(train_loader), 'recon_loss': train_loss_recon/len(train_loader), 'kl_loss': train_loss_kl/len(train_loader), 'triplet_loss': train_loss_triplet/len(train_loader)})
        test_losses.append({'total_loss': test_loss/len(test_loader), 'vae_loss': test_vae_loss/len(train_loader), 'recon_loss': test_loss_recon/len(test_loader), 'kl_loss': test_loss_kl/len(test_loader), 'triplet_loss': test_loss_triplet/len(test_loader)})
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs}')
            print("Training")
            print(f'Total Loss: {train_loss/len(train_loader)}')
            print(f'Total VAE Loss: {train_vae_loss/len(train_loader)}')
            print(f'Recon Loss: {train_loss_recon/len(train_loader)}, KL Loss: {train_loss_kl/len(train_loader)}, Triplet Loss: {train_loss_triplet/len(train_loader)}')

            print("Testing")
            print(f'Total Loss: {test_loss/len(test_loader)}')
            print(f'Total VAE Loss: {test_vae_loss/len(train_loader)}')
            print(f'Recon Loss: {test_loss_recon/len(test_loader)}, KL Loss: {test_loss_kl/len(test_loader)}, Triplet Loss: {test_loss_triplet/len(test_loader)}')
            print('------------------------------------\n')

        ## Track the best model
        if test_loss < best_loss:
            best_loss = test_loss
            ## Save the model temporarily
            print("**** Saving the model ***\n")
            torch.save(model.state_dict(), '../Dataset/Saved_models/temp_best_model.pth')
            es_counter = 0
            
        else:
            es_counter += 1
            if es_counter == 100:
                print('Early stopping')
                print(f'Best Test loss: {best_loss}')
                ## Load the best model
                model.load_state_dict(torch.load('../Dataset/Saved_models/temp_best_model.pth'))
                return model, best_loss, {'train': train_losses, 'test': test_losses}