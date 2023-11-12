import torch
import torch.nn as nn


class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=False, conditional_size=0, scaler=None):
        super().__init__()

        if conditional:
            assert conditional_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, conditional_size)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, conditional_size, scaler)


    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, means, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return means + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, conditional_size):
        super().__init__()

        self.weekconv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, stride=7),
                                      nn.ELU(), nn.Dropout(0.1),
                                      )
        self.dayconv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=24, stride=24),
                                     nn.ELU(), nn.Dropout(0.1),
                                     )
        self.hourconv = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=8, kernel_size=12, stride=12),
                                      nn.ELU(), nn.Dropout(0.1),
                                      )

        self.MLP = nn.Sequential(
            nn.Linear(4480, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, layer_sizes[-1]))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = x.unsqueeze(1)
        x_h = self.hourconv(x)
        x_h = x_h.reshape(x_h.size(0), 1, -1)
        x_d = self.dayconv(x_h)
        x_d = x_d.reshape(x_d.size(0), 1, -1)
        x_w = self.weekconv(x_d)
        x_w = x_d.reshape(x_w.size(0), 1, -1)
        x = torch.cat((x_w, x_d, x_h), dim=-1)
        x = self.MLP(x)

        means = self.linear_means(x)
        means = torch.squeeze(means, dim=1)
        log_vars = self.linear_log_var(x)
        log_vars = torch.squeeze(log_vars, dim=1)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_size, conditional, conditional_size, scaler):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + conditional_size
        else:
            input_size = latent_size

        hidden_size=128
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.1,
            # batch_first=True,
        )

        self.MLP = nn.Sequential(
                                 nn.Linear(input_size , input_size), nn.Dropout(0.1), nn.ELU(),
                                 nn.Linear(input_size, conditional_size),
                                 nn.Dropout(0.1), nn.ELU())

        self.scaler = scaler

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        z = z.unsqueeze(1)
        x, hidden_state = self.rnn(z)
        x = self.MLP(x)
        x = x.squeeze(1)

        return x

