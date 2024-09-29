import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image
import os
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "C:/Users/Keerthivasan/Desktop/Hackathons/archive/chest_xray/train"
image_size = 256
batch_size = 8

transform = T.Compose([
    T.Resize(image_size),
    T.CenterCrop(image_size),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_ds = ImageFolder(DATA_DIR, transform=transform)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)
latent_size = 256
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_size, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

generator = Generator().to(device)
discriminator = Discriminator().to(device)

adversarial_loss = torch.nn.BCEWithLogitsLoss()

lr = 0.0001
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

num_epochs = 100
sample_interval = 200

if __name__ == '__main__':

    if os.path.exists('generator.pth'):
        generator.load_state_dict(torch.load('generator.pth'))
        discriminator.load_state_dict(torch.load('discriminator.pth'))

    for epoch in range(num_epochs):
        for i, (imgs, _) in enumerate(train_dl):
            batch_size = imgs.size(0)
            valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)
            fake = torch.FloatTensor(batch_size, 1).fill_(0.0).to(device)
            real_imgs = imgs.to(device)

            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_size).to(device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid[:batch_size])
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake[:batch_size])
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_dl)}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")

            batches_done = epoch * len(train_dl) + i
            if batches_done % sample_interval == 0:
                os.makedirs('images', exist_ok=True)
                save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)

    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
