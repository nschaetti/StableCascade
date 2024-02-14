#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import yaml
import torch
from tqdm import tqdm

os.chdir('..')

from inference.utils import *
from train import WurstCoreC, WurstCoreB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# SETUP STAGE C
# Open Stage C config
# version, chechpoint paths
config_file = 'configs/inference/stage_c_3b.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)
# end with

# WurstCoreC is in train/train_c.py
core_c = WurstCoreC(
    config_dict=loaded_config,
    device=device,
    training=False
)

# SETUP STAGE B
# Stage B config
# batch size, image size, checkpoint paths
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
# end with

# WurstCoreB is in train/train_b.py
core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

# Setup sampling configuration, preprocessing and transforms
extras_c = core_c.setup_extras_pre()

# Create models and load checkpoints
models = core_c.setup_models(extras_c)

# Eval mode
models.generator.eval().requires_grad_(False)
print("Info:: Stage C loaded and ready")

# Setup sampling configuration, preprocessing and transforms
extras_b = core_b.setup_extras_pre()

# Create models and load checkpoints
models_b = core_b.setup_models(extras_b, skip_clip=True)

# Add the tokenizer and text model to Stage C
models_b = WurstCoreB.Models(
   **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
)

# Eval mode
models_b.generator.bfloat16().eval().requires_grad_(False)
print("Info:: Stage B loaded and ready")

# Compile generator
models = WurstCoreC.Models(
   **{
       **models.to_dict(),
       'generator': torch.compile(models.generator, mode="reduce-overhead", fullgraph=True)
   }
)

# Compile generator
models_b = WurstCoreB.Models(
   **{
       **models_b.to_dict(),
       'generator': torch.compile(models_b.generator, mode="reduce-overhead", fullgraph=True)
   }
)

# Batch size
batch_size = 4

# Prompt and image size
# caption = "Cinematic photo of an anthropomorphic nerdy rodent sitting in a cafe reading a book"
caption = "Cinematic photo of an anthropomorphic penguin sitting in a cafe reading a book and having a coffee"
height, width = 1024, 1024

# Calculate latent sizes
# stage_c_latent_shape: (batch_size, 16, latent_height, latent_width)
# stage_b_latent_shape: (batch_size, 4, latent_height, latent_width)
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras_c.sampling_configs['cfg'] = 4
extras_c.sampling_configs['shift'] = 2
extras_c.sampling_configs['timesteps'] = 20
extras_c.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# Prepare prompts for batches
batch = {'captions': [caption] * batch_size}

# Prepare conditions for stage C
conditions_c = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=False, eval_image_embeds=False)
unconditions_c = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=True, eval_image_embeds=False)

# Prepare conditions for stage B
conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

# No gradient, bfloat16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Sample stage C
    sampling_c = extras_c.gdf.sample(
        models.generator,
        conditions_c,
        stage_c_latent_shape,
        unconditions_c,
        device=device,
        **extras_c.sampling_configs
    )

    # Sample stage C
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
    # end for

    # Conditions for sample stage B
    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    # Sample stage B
    sampling_b = extras_b.gdf.sample(
        models_b.generator,
        conditions_b,
        stage_b_latent_shape,
        unconditions_b,
        device=device,
        **extras_b.sampling_configs
    )

    # Sample each step
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    # end for

    # Decode stage B to get the final image
    sampled = models_b.stage_a.decode(sampled_b).float()
# end with no grad, autocast

# Show the images
show_images(sampled)

# In[33]:


batch_size = 4
# url = "https://media.discordapp.net/attachments/1121232062708457508/1204613422680113212/1707272583_2.png?ex=65d55eac&is=65c2e9ac&hm=7741ea0f494b04c830128d883d7f03b28b5caffb85959d3fc0a0fb3d18d0411e&=&format=webp&quality=lossless"
url = "https://media.discordapp.net/attachments/1121232062708457508/1205134776206491648/image.png?ex=65d74438&is=65c4cf38&hm=fcb40fc6bbe437dee481afffcd94e25c5511d059341b2f2b6e046f157e6b9371&=&format=webp&quality=lossless"
images = resize_image(download_image(url)).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

batch = {'images': images}

show_images(batch['images'])


# In[34]:


caption = ""
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras_c.sampling_configs['cfg'] = 4
extras_c.sampling_configs['shift'] = 2
extras_c.sampling_configs['timesteps'] = 20
extras_c.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
    conditions = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=False, eval_image_embeds=True)
    unconditions = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=True, eval_image_embeds=False)
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    sampling_c = extras_c.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras_c.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
        
    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()

show_images(sampled)


# ## Image-to-Image

# In[20]:


batch_size = 4
url = "https://media.discordapp.net/attachments/1121232062708457508/1204557773480656947/chrome_rodent_knight.png?ex=65d52ad8&is=65c2b5d8&hm=8e74f16e685e54a4f67337fedbdfea350169d03babc6c2f79fd1b74a6fb665fb&=&format=webp&quality=lossless"
images = resize_image(download_image(url)).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

batch = {'images': images}

show_images(batch['images'])


# In[22]:


caption = "a person riding a rodent"
noise_level = 0.8
height, width = 1024, 1024
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

effnet_latents = core_c.encode_latents(batch, models, extras_c)
t = torch.ones(effnet_latents.size(0), device=device) * noise_level
noised = extras_c.gdf.diffuse(effnet_latents, t=t)[0]

# Stage C Parameters
extras_c.sampling_configs['cfg'] = 4
extras_c.sampling_configs['shift'] = 2
extras_c.sampling_configs['timesteps'] = int(20 * noise_level)
extras_c.sampling_configs['t_start'] = noise_level
extras_c.sampling_configs['x_init'] = noised

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)
    conditions = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core_c.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=True, eval_image_embeds=False)
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    sampling_c = extras_c.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras_c.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
        
    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()

show_images(batch['images'])
show_images(sampled)

