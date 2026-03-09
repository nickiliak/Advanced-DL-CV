import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
from torch import optim
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

from ddpm import Diffusion
from model import UNet

SEED = 1
DATASET_SIZE = 40000

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def prepare_dataloader(batch_size):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from dataset.sprites_dataset import SpritesDataset
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'exercise2.1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
    sprites_data = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config['sprites_data']))
    dataset = SpritesDataset(transform, img_file=sprites_data, num_samples=config.get('num_samples', DATASET_SIZE), seed=config.get('seed', SEED))
    dataloader = DataLoader(dataset, batch_size=config.get('batch_size', batch_size), shuffle=True)
    return dataloader

def create_result_folders(experiment_name):
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'exercise2.1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.abspath(os.path.join(base_dir, config['models_dir']))
    results_dir = os.path.abspath(os.path.join(base_dir, config['outputs_dir']))
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, experiment_name), exist_ok=True)
    os.makedirs(os.path.join(results_dir, experiment_name), exist_ok=True)

def train(device='cpu', T=500, img_size=16, input_channels=3, channels=32, time_dim=256,
          batch_size=100, lr=1e-3, num_epochs=30, experiment_name="ddpm", show=False):
    """Implements algrorithm 1 (Training) from the ddpm paper at page 4"""
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'exercise2.1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    create_result_folders(experiment_name)
    dataloader = prepare_dataloader(batch_size)

    model = UNet(img_size=config.get('img_size', img_size), c_in=config.get('input_channels', input_channels), c_out=config.get('input_channels', input_channels), 
                 time_dim=config.get('time_dim', time_dim),channels=config.get('channels', channels), device=device).to(device)
    diffusion = Diffusion(img_size=config.get('img_size', img_size), T=config.get('T', T), beta_start=config.get('beta_start', 1e-4), beta_end=config.get('beta_end', 0.02), device=device)

    optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', lr))
    mse = ... # use MSE loss 
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(base_dir, 'outputs', 'runs')
    logger = SummaryWriter(os.path.join(runs_dir, experiment_name))
    l = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        for i, images in enumerate(pbar):
            images = images.to(device)

            # TASK 4: implement the training loop
            t = diffusion.sample_timesteps(images.shape[0]).to(device) # line 3 from the Training algorithm
            x_t, noise = ... # inject noise to the images (forward process), HINT: use q_sample
            predicted_noise = ... # predict noise of x_t using the UNet
            loss = ... # loss between noise and predicted noise

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.p_sample_loop(model, batch_size=images.shape[0])
        results_dir = os.path.join(os.path.abspath(os.path.join(base_dir, config['outputs_dir'])), experiment_name)
        models_dir = os.path.join(os.path.abspath(os.path.join(base_dir, config['models_dir'])), experiment_name)
        save_images(images=sampled_images, path=os.path.join(results_dir, f"{epoch}.jpg"),
            show=show, title=f'Epoch {epoch}')
        torch.save(model.state_dict(), os.path.join(models_dir, f"weights-{epoch}.pt"))


def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'exercise2.1.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=config.get('seed', SEED))
    train(device=device)

if __name__ == '__main__':
    main()
    

        