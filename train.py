from data import get_dataloader
from models import VAE
from utils import vae_kl_loss, show_images
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR = 1e-3
EPOCHES = 10
factor = 1000
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()
loader = get_dataloader()


model.train()
iter = 0
for epoch in range(EPOCHES):
    for x, _ in loader:
        y, mu, log_var = model(x)
        vae_loss = criterion(y, x)
        kl_loss = vae_kl_loss(mu, log_var)
        total_loss = factor * vae_loss + kl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if iter % 500 == 0:
            print('iter:{}\tvae_loss:{:.3f}\tkl_loss:{:.3f}'.format(iter, vae_loss.item(), kl_loss.item()))
            with torch.no_grad():
                decoder = model.decoder
                decoder.eval()
                z = torch.randn(64, 4, device=device)
                decoder.train()
                fake_images = decoder(z).data.cpu()
                show_images(fake_images)
        iter += 1




