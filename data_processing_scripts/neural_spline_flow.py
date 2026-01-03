# Import required packages
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def setup_data_for_train(train_samples, test_samples, device='cpu', batch_size=512, shuffle=True):
    
    # Convert the DataFrame to a PyTorch Tensor
    train_tensor = torch.tensor(train_samples, dtype=torch.float32).to(device)
    test_tensor = torch.tensor(test_samples, dtype=torch.float32).to(device)

    # Create a TensorDataset
    train_dataset = TensorDataset(train_tensor)

    # Create a DataLoader with a batch size of 512
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"DataLoader created for two_modal_samples_df with batch size {batch_size}.")
    print(f"Number of batches: {len(train_dataloader)}")

    return train_dataloader, test_tensor


def setup_model(latent_size, K=16, seed=42, hidden_units=128, hidden_layers=2, trainable=False, device='cpu'):
    torch.manual_seed(seed)

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=trainable)
    
    # Construct flow model
    model = nf.NormalizingFlow(q0=q0, flows=flows)
    model.to(device)

    return model


def train_loop(model, train_dataloader, save_prefix, max_iter=4000, show_iter=500, lr=5e-4, weight_decay=1e-5):

    # Train model
    loss_hist = np.array([])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create an iterator for the dataloader
    data_iter = iter(train_dataloader)

    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        try:
            # Get training samples from the dataloader
            # Dataloader yields (tensor_data,), so unpack with a comma
            x, = next(data_iter)
        except StopIteration:
            # Reset the iterator if it's exhausted
            data_iter = iter(train_dataloader)
            x, = next(data_iter)

    # x is already on device due to prior tensor creation

    # Compute loss
    loss = model.forward_kld(x)

    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

    # Plot loss
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.savefig(f"./models/{save_prefix}_loss_hist.png")

    return model
    











