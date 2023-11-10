import torch
from inference.inference import Inference
from PIL import Image
from RealESRGAN import RealESRGAN
from diffusers.utils import load_image
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image


def resolution(image, height, width, device):
    model = RealESRGAN(device, scale=4)
    model.load_weights("weights/RealESRGAN_x4.pth", download=True)

    # resolution image
    sr_image = model.predict(image)
    sr_image = sr_image.resize((width, height))

    return sr_image


def refiner(
    image,
    prompt,
    device,
):
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    image_refiner = pipe(prompt, image=image).images[0]
    return image_refiner
