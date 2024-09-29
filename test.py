import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_size = 256
image_size = 256

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

generator = Generator().to(device)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

def generate_and_save_images(num_images=100, batch_size=16, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():  
        for i in range(0, num_images, batch_size):
            z = torch.randn(min(batch_size, num_images - i), latent_size).to(device)  
            gen_imgs = generator(z)  
            
            for j in range(gen_imgs.size(0)):
                save_image(gen_imgs[j], f"{output_dir}/generated_image_{i + j + 1}.png", normalize=True)
            print(f"Images {i + 1} to {i + gen_imgs.size(0)} saved to {output_dir}")

if __name__ == '__main__':
    generate_and_save_images(num_images=5, batch_size=16)
