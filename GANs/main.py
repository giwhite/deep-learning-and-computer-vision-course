import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

def train_gan(generator, discriminator, dataloader, num_epochs, batch_size, latent_dim, device):
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0001)

    fixed_noise = torch.randn(64, latent_dim, device=device)

    for epoch in range(num_epochs):
        acc = 0
        num = 0

        for batch_data in tqdm(dataloader):
            real_data = batch_data[0].view(batch_size, -1).to(device)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            optimizer_D.zero_grad()
            fake_data = generator(torch.randn(batch_size, latent_dim, device=device))          
            real_logtis = discriminator(real_data)
            fake_logtis = discriminator(fake_data)

            real_loss = criterion(real_logtis, real_labels)
            fake_loss = criterion(fake_logtis, fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
        
            optimizer_G.zero_grad()
            fake_data = generator(torch.randn(batch_size, latent_dim, device=device))
            g_loss = criterion(discriminator(fake_data), real_labels)
            g_loss.backward()
            optimizer_G.step()


            #test
            tp = (real_logtis>0.5).sum()
            fn = (fake_logtis<0.5).sum()
            num += 2*batch_size
            acc += tp + fn

        print(f"Epoch [{epoch+1}/{num_epochs}]| Discriminator Loss: {d_loss.item()} | Generator Loss: {g_loss.item()}| Accuracy is {acc/num} ")


        if (epoch+1) % 10 == 1:
            generate_and_save_images(generator, epoch + 1, fixed_noise)
            save_model(generator,epoch)


def save_model(generator,epoch,file_path='./model/model.pth'):
    torch.save(generator.state_dict(), file_path)
    print(f"Generator model saved to {file_path} in {epoch}")



def load_model(generator,file_path='./model/model.pth'):
    generator.load_state_dict(torch.load(file_path))
    print(f"Generator model loaded from {file_path}")


def generate_and_save_images(generator, epoch, test_input, save_name=''):
    with torch.no_grad():
        generated_images = generator(test_input).cpu().detach()
    generated_images = generated_images.view(-1, 28, 28).numpy()
    plt.figure(figsize=(8, 8))
    for i in range(generated_images.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(generated_images[i], cmap='gray')
        plt.axis('off')
    if save_name=='':
        save_name = f"./img/gan_generated_image_epoch_{epoch}.png"
    plt.savefig(save_name)
    return generated_images
    #plt.show()


if __name__ =="__main__":
    device = torch.device("cuda:1")
    batch_size = 32
    # 数据预处理和加载
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化生成器和判别器
    latent_dim = 100
    img_dim = 28 * 28
    generator = Generator(latent_dim, img_dim).to(device)
    discriminator = Discriminator(img_dim).to(device)

    # 训练GAN
    num_epochs = 1000
    load_m = False
    if load_m:
        load_model(generator, "./model/model_100.pth")
    train_gan(generator, discriminator, dataloader, num_epochs, batch_size, latent_dim, device)

