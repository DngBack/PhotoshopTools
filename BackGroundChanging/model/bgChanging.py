from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import numpy as np
from comet_ml import Experiment


def ChangingBg(image, mask_image, prompt, generator):
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    image_out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]

    return image_out


def ChangingBg_refiner(image, mask_image, prompt, device):
    sd_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)

    refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=sd_pipe.text_encoder_2,
        vae=sd_pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    exp = Experiment()
    exp.set_model_graph([sd_pipe, refiner])
    pass
