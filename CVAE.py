import torch
import torch.nn as nn

#https://graviraja.github.io/conditionalvae/
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: An integer indicating the size of input.
            hidden_dim: An integer indicating the size of hidden dimension.
            latent_dim: An integer indicating the latent size.
        '''
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout())
        self.mu = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU(), nn.Dropout())
        self.var = nn.Sequential(nn.Linear(hidden_dim, latent_dim), nn.ReLU(), nn.Dropout())

    def forward(self, x):
        # x is of shape [batch_size, input_dim]

        hidden = self.linear(x)
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var

class AutoDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        '''
        Args:
            latent_dim: An integer indicating the latent size.
            hidden_dim: An integer indicating the size of hidden dimension.
            output_dim: An integer indicating the size of output.
        '''
        super().__init__()

        self.latent_to_hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout())
        self.hidden_to_out = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU(), nn.Dropout())

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        x = self.latent_to_hidden(x)
        # x is of shape [batch_size, hidden_dim]
        generated_x = self.hidden_to_out(x)
        # x is of shape [batch_size, output_dim]

        return generated_x

class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: An integer indicating the size of input.
            hidden_dim: An integer indicating the size of hidden dimension.
            latent_dim: An integer indicating the latent size.
        '''
        super().__init__()

        self.encoder = AutoEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = AutoDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        generated_x = self.decoder(x_sample)

        return generated_x, z_mu, z_var