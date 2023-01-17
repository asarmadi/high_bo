import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalEncoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1  = nn.Linear(in_dim, in_dim//2)
        self.linear11 = nn.Linear(in_dim//2, in_dim//4)
        self.linear2  = nn.Linear(in_dim//4, latent_dims)
        self.linear3  = nn.Linear(in_dim//4, latent_dims)
        self.N       = torch.distributions.Normal(0, 1)
        self.N.loc   = self.N.loc
        self.N.scale = self.N.scale
        self.kl      = 0

    def forward(self, x):
        x       = F.relu(self.linear1(x))
        x       = F.relu(self.linear11(x))
        mu      = self.linear2(x)
        sigma   = self.linear3(x)
        std     = torch.exp(0.5*sigma)
        eps     = torch.randn_like(sigma)
#        z       = mu + sigma*self.N.sample(mu.shape)
        z       = mu + std*eps
#        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = torch.mean(-0.5 * torch.sum(1 + sigma - mu ** 2 - sigma.exp(), dim = 1), dim = 0)
        return z

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim//2)
        self.linear2 = nn.Linear(in_dim//2, in_dim//2)
        self.linear3 = nn.Linear(in_dim//2, latent_dims)

    def forward(self, x):
        x       = F.relu(self.linear1(x))
        x       = F.relu(self.linear2(x))
        z       = F.sigmoid(self.linear3(x))
        return z

class Decoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(Decoder, self).__init__()
        self.linear1  = nn.Linear(latent_dims, in_dim//2)
        self.linear11 = nn.Linear(in_dim//2, in_dim//2)
        self.linear2  = nn.Linear(in_dim//2, in_dim)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear11(z))
        z = torch.sigmoid(self.linear2(z))
        return z

class Autoencoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(in_dim, latent_dims)
        self.decoder = Decoder(in_dim, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class VariationalAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_dim, latent_dims)
        self.decoder = Decoder(in_dim, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class CondVariationalAutoencoder(nn.Module):
    def __init__(self, in_dim, latent_dims):
        super(CondVariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_dim+1, latent_dims)
        self.decoder = Decoder(in_dim, latent_dims+1)

    def forward(self, x, c):
        X = torch.cat([x, c], 1)
        z = self.encoder(X)
        Z = torch.cat([z, c], 1)
        return self.decoder(Z)
