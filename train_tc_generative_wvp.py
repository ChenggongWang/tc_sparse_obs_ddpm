import os
import xarray as xr
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.optim import Adam 
from einops import rearrange
import tqdm

from denoising_diffusion_pytorch import Unet
from tc_data_loader import TC_xy_Dataset 
from tc_diff_guided import Diffusion_model, count_parameters
def save_model(model,name,results_folder):
    torch.save(model.state_dict(), results_folder/f'{name}_model.ckpt')
    
def compute_val_loss(tc_diff_model, val_loader, obs_ratio):
    loss_val = []
    tc_diff_model.model.eval()
    with torch.no_grad():
        # evaluate model on 1024 samples in valiation data
        for step, batch in enumerate(val_loader): 
            batch_size = batch.shape[0]
            batch = batch.to(tc_diff_model.device)
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, tc_diff_model.timesteps, (batch_size,), device=tc_diff_model.device).long()
            loss = tc_diff_model.p_losses(tc_diff_model.model, batch, t, loss_type="huber", obs_ratio=obs_ratio)
            loss_val.append(loss.item())
            if step >1024/8:   break
    return np.mean(loss_val)
def train(tc_diff_model, dataloader, validation_data,val_loader, obs_ratio, results_folder, epochs):
    loss_train_all = []
    loss_valid_all = []
    loss_best = 100
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            tc_diff_model.model.train()
            tc_diff_model.optimizer.zero_grad()
    
            batch_size = batch.shape[0]
            batch = batch.to(tc_diff_model.device)
    
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, tc_diff_model.timesteps, (batch_size,), device=tc_diff_model.device).long()
            loss = tc_diff_model.p_losses(tc_diff_model.model, batch, t, loss_type="huber", obs_ratio=obs_ratio)
            if step % 200 == 0:
                loss_val = compute_val_loss(tc_diff_model, val_loader, obs_ratio)
                print(f"epoch: {epoch:02d} |step: {step:03d} | Train Loss: {loss.item():7.4f}| Val Loss: {loss_val:7.4f}" )
                loss_train_all.append(loss.item())
                loss_valid_all.append(loss_val)
                # save model if loss_val is smaller
                if loss_val < 0.9*loss_best:
                    loss_best = loss_val
                    save_model(tc_diff_model.model,'best',results_folder)
                
            loss.backward()
            tc_diff_model.optimizer.step()
    
        #save generated images  each epoch
        milestone = epoch
        all_images = tc_diff_model.sample(tc_diff_model.model, tc_diff_model.image_size, batch_size=36, 
                                        channels=tc_diff_model.channels, obs_ratio=obs_ratio,
                                        dataset=validation_data)[0][-1]
        all_images = (all_images + 1) * 0.5
        all_images = make_grid(torch.tensor(all_images), nrow = 6)
        save_image(all_images, str(results_folder / f'sample-{milestone}.png'))
    
    save_model(tc_diff_model.model,'final',results_folder)
    return loss_train_all, loss_valid_all

def main():
    timesteps = 300
    batch_size = 8
    results_folder = Path("./tc_guide_wvp")
    os.makedirs(results_folder, exist_ok=True)
    obs_ratio = 0.2


    training_data  = TC_xy_Dataset(data_dir='./dataset/tc_data',
                                   data_vars=['WVP'],
                                   years=[120,121,122,123,124]) 
    validation_data  = TC_xy_Dataset(data_dir='./dataset/tc_data',
                                   data_vars=['WVP'],
                                   years=[125]) 
    test_data  = TC_xy_Dataset(data_dir='./dataset/tc_data',
                                   data_vars=['WVP'],
                                   years=[126,127]) 

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    dataloader = train_loader
    val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True) 

    # diffusion model
    tc_diff_model = Diffusion_model(timesteps)

    # decoder model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = training_data.data.shape[-1]
    channels = training_data.data.shape[1]
    
    model = Unet(
        dim = 32,
        out_dim=channels,
        channels=channels*2, 
        dim_mults = (1,2,4), 
        resnet_block_groups = 4,
        flash_attn = False
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    print('Total model parameter',count_parameters(model))

    tc_diff_model.model = model
    tc_diff_model.device = device
    tc_diff_model.optimizer = optimizer
    tc_diff_model.image_size = image_size
    tc_diff_model.channels = channels
    train(tc_diff_model, 
          dataloader, validation_data, val_loader, obs_ratio, 
          results_folder, epochs = 4)
    
    print('trainning donw. model are saved at: ', results_folder)
if __name__ == "__main__":
    main()