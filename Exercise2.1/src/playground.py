import yaml
import os
import numpy as np
import matplotlib.pyplot as plt

# torch imports
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

# custom imports
from ddpm import Diffusion
from model import UNet

from dataset.helpers import im_normalize, tens2image
from dataset import SpritesDataset

def show(imgs, title=None, fig_titles=None, save_path=None): 

    if fig_titles is not None:
        assert len(imgs) == len(fig_titles)

    fig, axs = plt.subplots(1, ncols=len(imgs), figsize=(15, 5))
    for i, img in enumerate(imgs):
        axs[i].imshow(img)
        axs[i].axis('off')
        if fig_titles is not None:
            axs[i].set_title(fig_titles[i])

    if title is not None:
        plt.suptitle(title)
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    plt.show()

if __name__ == '__main__':
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'exercise2.1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 2929)
    torch.manual_seed(seed)

    outputs = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config['outputs_dir']))
    os.makedirs(outputs, exist_ok=True)

    # dataset and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])

    batch_size = config.get('batch_size', 8)
    sprites_data = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config['sprites_data']))
    trainset = SpritesDataset(transform=transform, img_file=sprites_data, num_samples=config.get('num_samples', 40000), seed=seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    images  = next(iter(trainloader))
    # visualize examples
    example_images = np.stack([im_normalize(tens2image(images[idx])) for idx in range(batch_size)], axis=0)
    show(example_images, 'Example sprites', save_path=os.path.join(outputs, 'example.png'))

    ################## Diffusion class ##################
    # TASK 1: Implement beta, alpha, and alpha_hat 
    diffusion = Diffusion(device=device)
    plt.figure()
    plt.plot(range(1,diffusion.T+1), diffusion.alphas.cpu().numpy(), label='alphas', linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.alphas_bar.cpu().numpy(), label='alphas_bar',linewidth=3)
    plt.plot(range(1,diffusion.T+1), diffusion.betas.cpu().numpy(), label='betas', linewidth=3)
    plt.title('Diffusion parameters')
    plt.legend()
    plt.savefig(os.path.join(outputs, 'diffusion_params.png'), bbox_inches='tight')
    plt.show()
    #####################################################
    

    # timesteps for forward
    t = torch.Tensor([0, 50, 100, 150, 200, 300, 499]).long().to(device)
    fig_titles = [f'Step {ti.item()}' for ti in t]
    x0 = images[0].unsqueeze(0).to(device) # add batch dimenstion

    ################## Forward process ##################
    # TASK 2: Implement it in the diffusion class
    xt, noise = diffusion.q_sample(x0, t)
    #####################################################

    noised_images = np.stack([im_normalize(tens2image(xt[idx].cpu())) for idx in range(t.shape[0])], axis=0)
    show(noised_images, title='Forward process', fig_titles=fig_titles, save_path=os.path.join(outputs, 'forward.png'))

    ################## Inverse process ##################
    model = UNet(device=device)
    model.eval()
    model.to(device)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'))
    model_path = os.path.join(models_dir, 'weights-59epochs-full-dataset.pt')
    model.load_state_dict(torch.load(model_path, map_location=device)) # load the given model
    torch.manual_seed(seed)

    # TASK 3: Implement it in the diffusion class
    x_new, intermediate_images = diffusion.p_sample_loop(model, 1, timesteps_to_save=t)
    intermediate_images = [tens2image(img.cpu()) for img in intermediate_images]
    show(intermediate_images, title='Reverse process', fig_titles=fig_titles, save_path=os.path.join(outputs, 'reverse.png'))
    #####################################################

