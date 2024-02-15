#!/usr/bin/env python
# coding: utf-8

import os
import yaml
from tqdm import tqdm
from inference.utils import *
from train import WurstCoreC, WurstCoreB


# Get device
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
core_b = WurstCoreB(
    config_dict=config_file_b,
    device=device,
    training=False
)

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

# Compile generator
# models = WurstCoreC.Models(
#     **{
#         **models_c.to_dict(),
#         'generator': torch.compile(models_c.generator, mode="reduce-overhead", fullgraph=True)
#     }
# )

# Compile generator
# models_b = WurstCoreB.Models(
#     **{
#         **models_b.to_dict(),
#         'generator': torch.compile(models_b.generator, mode="reduce-overhead", fullgraph=True)
#     }
# )

# Batch size
batch_size = 4

# end of common code

# Prompt and image size
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
conditions_c = core_c.get_conditions(
    batch,
    models_c,
    extras_c,
    is_eval=True,
    is_unconditional=False,
    eval_image_embeds=False
)

# Prepare unconditions for stage C
unconditions_c = core_c.get_conditions(
    batch,
    models_c,
    extras_c,
    is_eval=True,
    is_unconditional=True,
    eval_image_embeds=False
)

# Prepare conditions for stage B
conditions_b = core_b.get_conditions(
    batch,
    models_b,
    extras_b,
    is_eval=True,
    is_unconditional=False
)

# Prepare unconditions for stage B
unconditions_b = core_b.get_conditions(
    batch,
    models_b,
    extras_b,
    is_eval=True,
    is_unconditional=True
)

# No gradient, bfloat16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Sample stage C
    sampling_c = extras_c.gdf.sample(
        models_c.generator,
        conditions_c,
        stage_c_latent_shape,
        unconditions_c,
        device=device,
        **extras_c.sampling_configs
    )

    # Sample stage C
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
    # end forx§

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
image_grid = show_images(sampled, return_images=True)
image_grid.save("image_generation.png")
