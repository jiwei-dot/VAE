import torch
import torch.nn as nn


class Conv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_BN_ReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class TransposeConv_BN_ReLU(nn.Module):
    def __init__(self, in_channels, out_channels, output_padding, stride):
        super(TransposeConv_BN_ReLU, self).__init__()
        self.tconv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, output_padding,
                                        bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.block1 = Conv_BN_ReLU(1, 32, 1)
        self.block2 = Conv_BN_ReLU(32, 64, 2)
        self.block3 = Conv_BN_ReLU(64, 64, 2)
        self.block4 = Conv_BN_ReLU(64, 64, 1)
        self.fc1 = nn.Linear(3136, 4)
        self.fc2 = nn.Linear(3136, 4)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var


class Sample(nn.Module):
    def forward(self, mu, log_var):
        eplison = torch.rand_like(mu)
        return eplison * torch.exp((log_var / 2)) + mu


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(4, 3136)
        self.block1 = TransposeConv_BN_ReLU(64, 64, 0, 1)
        self.block2 = TransposeConv_BN_ReLU(64, 64, 1, 2)
        self.block3 = TransposeConv_BN_ReLU(64, 32, 1, 2)
        self.tconv = nn.ConvTranspose2d(32, 1, 3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, 64, 7, 7)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.tconv(x)
        out = self.sigmoid(x)
        return out
    
    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.sample = Sample()
        self.decoder = Decoder()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        x = self.sample(mu, log_var)
        out = self.decoder(x)
        return out, mu, log_var


if __name__ == '__main__':
    from torchsummary import summary
    import torch
    x = torch.randn(32, 1, 28, 28)
    vae = VAE()
    print(vae(x).shape)