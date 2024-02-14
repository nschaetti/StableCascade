#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import yaml
from tqdm import tqdm

os.chdir('..')

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
models = WurstCoreC.Models(
    **{
        **models_c.to_dict(),
        'generator': torch.compile(models_c.generator, mode="reduce-overhead", fullgraph=True)
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

# end of common code

# URL
url = "https://media.discordapp.net/attachments/1121232062708457508/1205134776206491648/image.png?ex=65d74438&is=65c4cf38&hm=fcb40fc6bbe437dee481afffcd94e25c5511d059341b2f2b6e046f157e6b9371&=&format=webp&quality=lossless"

# Resize images
# and multiply for batch size
images = resize_image(
    download_image(url)
).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

# Source images as a batch
batch = {'images': images}

# Show images
show_images(batch['images'])

# No caption, image size
caption = ""
# There is no noise_level here (compare to image to image)
height, width = 1024, 1024

# Calculate latent sizes
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Compared to image to image
# We don't encode the batch to latents
# We don't add noise

# Stage C Parameters
extras_c.sampling_configs['cfg'] = 4
extras_c.sampling_configs['shift'] = 2
extras_c.sampling_configs['timesteps'] = 20
extras_c.sampling_configs['t_start'] = 1.0
# Compare to image to image, we don't set x_init (initial state)

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# Prepare prompt for batches
# Prompt is empty here
batch['captions'] = [caption] * batch_size

# With no gradient, bfloat16
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Condition for Stage C
    # Returns a dictionary with the keys: 'clip_text', 'clip_text_pooled', 'clip_img'
    conditions_c = core_c.get_conditions(
        batch=batch,
        models=models,
        extras=extras_c,
        is_eval=True,
        is_unconditional=False,
        eval_image_embeds=True
        # For img2img, eval_image_embeds is False
    )

    # Negative condition for Stage C
    # Uncondition is empty prompt, with zero image embeds
    unconditions_c = core_c.get_conditions(
        batch=batch,
        models=models,
        extras=extras_c,
        is_eval=True,
        is_unconditional=True,
        eval_image_embeds=False
        # For img2img, eval_image_embeds is False
    )

    # Condition for Stage B
    # Returns a dictionary with the keys: 'effnet', 'clip'
    conditions_b = core_b.get_conditions(
        batch=batch,
        models=models_b,
        extras=extras_b,
        is_eval=True,
        is_unconditional=False
    )

    # Negative condition for Stage B
    # Uncondition is empty prompt, with zero image embeds
    unconditions_b = core_b.get_conditions(
        batch=batch,
        models=models_b,
        extras=extras_b,
        is_eval=True,
        is_unconditional=True
    )

    # Sampler for stage C
    sampling_c = extras_c.gdf.sample(
        models.generator,
        conditions_c,
        stage_c_latent_shape,
        unconditions_c,
        device=device,
        **extras_c.sampling_configs,
    )

    # Generate samples
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
        sampled_c = sampled_c
    # end for

    # Conditions for stage B
    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    # Sampler for stage B
    sampling_b = extras_b.gdf.sample(
        model=models_b.generator,
        model_inputs=conditions_b,
        shape=stage_b_latent_shape,
        unconditional_inputs=unconditions_b,
        device=device,
        **extras_b.sampling_configs
    )

    # For each step
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    # end for

    # Decode to get the final image
    sampled = models_b.stage_a.decode(sampled_b).float()
# end with no grad, autocast

# Show images
show_images(sampled)
