#!/usr/bin/env python
# coding: utf-8


# Imports
import os
import yaml
from tqdm import tqdm


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
models_c = core_c.setup_models(extras_c)

# Eval mode
models_c.generator.eval().requires_grad_(False)
print("Info:: Stage C loaded and ready")

# Setup sampling configuration, preprocessing and transforms
extras_b = core_b.setup_extras_pre()

# Create models and load checkpoints
models_b = core_b.setup_models(extras_b, skip_clip=True)

# Add the tokenizer and text model to Stage C
models_b = WurstCoreB.Models(
    **{**models_b.to_dict(), 'tokenizer': models_c.tokenizer, 'text_model': models_c.text_model}
)

# Eval mode
models_b.generator.bfloat16().eval().requires_grad_(False)
print("Info:: Stage B loaded and ready")

# # Compile generator
# models_c = WurstCoreC.Models(
#     **{
#         **models_c.to_dict(),
#         'generator': torch.compile(models_c.generator, mode="reduce-overhead", fullgraph=True)
#     }
# )
#
# # Compile generator
# models_b = WurstCoreB.Models(
#     **{
#         **models_b.to_dict(),
#         'generator': torch.compile(models_b.generator, mode="reduce-overhead", fullgraph=True)
#     }
# )

# Batch size
batch_size = 4

# end of common code

# Image URL
url = "https://media.discordapp.net/attachments/1121232062708457508/1204557773480656947/chrome_rodent_knight.png?ex=65d52ad8&is=65c2b5d8&hm=8e74f16e685e54a4f67337fedbdfea350169d03babc6c2f79fd1b74a6fb665fb&=&format=webp&quality=lossless"

# Resize the image and muliply for batch
images = resize_image(
    download_image(url)
).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

# Source images as a batch
batch = {'images': images}

# Show the images
show_images(batch['images'])

# Prompt, noise and image sizes
caption = "A reptile riding a blue fluffy cat"
noise_level = 0.8
height, width = 1024, 1024

# Calculate latent sizes
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Encode images to latents
effnet_latents = core_c.encode_latents(batch, models_c, extras_c)

# Add noise
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

# Prepare prompts for batches
batch['captions'] = [caption] * batch_size

# With no gradient, bfloat16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Stage C conditions
    conditions = core_c.get_conditions(
        batch=batch,
        models=models_c,
        extras=extras_c,
        is_eval=True,
        is_unconditional=False,
        eval_image_embeds=False
    )

    # Stage C unconditions
    unconditions = core_c.get_conditions(
        batch=batch,
        models=models_c,
        extras=extras_c,
        is_eval=True,
        is_unconditional=True,
        eval_image_embeds=False
    )

    # Stage B conditions
    conditions_b = core_b.get_conditions(
        batch=batch,
        models=models_b,
        extras=extras_b,
        is_eval=True,
        is_unconditional=False
    )

    # Stage B unconditions
    unconditions_b = core_b.get_conditions(
        batch,
        models_b,
        extras_b,
        is_eval=True,
        is_unconditional=True
    )

    # Sample stage C
    sampling_c = extras_c.gdf.sample(
        models_c.generator,
        conditions,
        stage_c_latent_shape,
        unconditions,
        device=device,
        **extras_c.sampling_configs,
    )

    # For each step
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
    # end for


    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator,
        conditions_b,
        stage_b_latent_shape,
        unconditions_b,
        device=device,
        **extras_b.sampling_configs
    )

    # For each step
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    # end for

    # Decode to get the final images
    sampled = models_b.stage_a.decode(sampled_b).float()
# end with no grad, autocast

# Show generated images
original_image_grid = show_images(batch['images'], return_images=True)
image_grid = show_images(sampled, return_images=True)
original_image_grid.save("original_image.png")
image_grid.save("image_to_image4.png")
# show_images(batch['images'])
# show_images(sampled)

