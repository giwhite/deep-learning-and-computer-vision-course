from main import load_model,generate_and_save_images,Generator
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
# 使用示例
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
generator = Generator(100, 28*28) # 创建一个新的生成器模型
load_model(generator, "./model/model_100.pth")  # 加载保存的生成器模型权重
fixed_noise = torch.randn(64, 100)
generated_images = generate_and_save_images(generator, 0, fixed_noise,'show_number')


num1 = fixed_noise[27].unsqueeze(0)
num2 = fixed_noise[49].unsqueeze(0)
all_ten = []
for i in range(8):
    all_ten.append((num2*i+num1*(7-i))/7)
all_ten.append(fixed_noise[:52])

c = torch.cat(all_ten,dim=0)

generate_and_save_images(generator, 0, fixed_noise[:8],'linerly_interpolating')


# label = [i[1] for i in dataset]
# count = [0]*10
# for i in label:
#     count[i] += 1
# print(count)

def find_sim_img():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    def MSE(img1,img2):
        mse = np.mean( (img1 - img2) ** 2 )
        return mse
    sim = []
    for j in tqdm(generated_images):
        l = []
        for i in dataset:
            diff = MSE(j,i[0].numpy())
            l.append(diff)
        sim.append(np.argmin(l))
    res = [dataset[i][0] for i in sim]
    res = torch.cat(res)
    res = res.view(-1, 28, 28).numpy()
    plt.figure(figsize=(8, 8))
    for i in range(res.shape[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(res[i], cmap='gray')
        plt.axis('off')
    plt.savefig("sim_img")


#find_sim_img()