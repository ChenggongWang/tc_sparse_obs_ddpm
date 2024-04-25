import torch 
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from tqdm.auto import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
class Diffusion_model():
    def __init__(self, timesteps, beta_schedule=linear_beta_schedule): 
         # define beta schedule
        betas = beta_schedule(timesteps=timesteps)
        
        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.timesteps = timesteps
        self.betas = betas
        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
        self.sqrt_recip_alphas = sqrt_recip_alphas
        self.posterior_variance = posterior_variance

        # initalize later
        self.model = None
        self.optimizer = None
        self.image_size = None
        self.channels = None
        self.device = None
        
    # Algorithm 1
    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
    
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
    
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1",obs_ratio=0.1):
        if noise is None:
            noise = torch.randn_like(x_start)
    
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy, x_cond = sparse_condition_rand(x_start,obs_ratio=obs_ratio,x_noise=x_noisy)
        predicted_noise = denoise_model(x_noisy, t)
    
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
    
        return loss
    # Algorithm 2
    # reverse diffusion from predicted noise
    # see derivation here
    # https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, obs_ratio, x_cond=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        # add sparse obs.
        x_new, x_cond = sparse_condition_rand(x, obs_ratio=obs_ratio, x_cond=x_cond)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x_new, t) / sqrt_one_minus_alphas_cumprod_t
        )
    
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
    # returning all images
    @torch.no_grad()
    def p_sample_loop(self, model, shape, x, obs_ratio):
        device = next(model.parameters()).device
    
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        # add fixed sparse obs. for the entire denoise process 
        x_new, x_cond = sparse_condition_rand(x.to(device), obs_ratio=obs_ratio) 
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), 
                                i, obs_ratio, x_cond = x_cond.to(device))
            imgs.append(img.cpu().numpy())
        return imgs
    
    @torch.no_grad()
    def sample(self, model, image_size, batch_size, channels, obs_ratio, dataset): 
        data_ind = np.random.randint(0,len(dataset),batch_size)
        data = [dataset[ind] for ind in data_ind] # select ind
        data = torch.tensor(np.array(data)) 
        return self.p_sample_loop(
                    model,
                    shape=(batch_size, channels, image_size, image_size),
                    x=data,
                    obs_ratio=obs_ratio
                ), data
        
# add conditional feature
def sparse_condition_rand(x, obs_ratio=0.1, x_noise=None, x_cond=None):
    if x_cond is None:
        # select random obs. as condition
        total_element = x.numel()
        x_cond = torch.zeros_like(x) 
        obs_num = np.floor(total_element*obs_ratio).astype(int)
        x_ind_rand = np.random.randint(0,total_element,obs_num) 
        x_cond.flatten()[x_ind_rand] = x.flatten()[x_ind_rand]
        
    x_noise = default(x_noise,x)
    if x.dim()==3:
        x_new = torch.cat([x_noise,x_cond],dim=0)
    elif x.dim()==4:
        x_new = torch.cat([x_noise,x_cond],dim=1)
    else:
        raise Exception('Please define concat dimension')
    return x_new, x_cond