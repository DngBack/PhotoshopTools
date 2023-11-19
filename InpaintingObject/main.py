import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
from diffusers import DiffusionPipeline
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import argparse
from inpaint_gen import InpaintingGenerativeV2
from diffusers import StableDiffusionInpaintPipeline

# Argument
parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate image")
parser.add_argument("-n", "--negative_prompt", type=str, help="Negative prompt")
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)


args = parser.parse_args()

# Setup hyper parameters
hyper_params = {
    "seed": 116,
    "kernel_size": (5, 5),
    "kernel_iterations": 15,
    "num_inference_steps": 70,
    "denoising_start": 0.70,
    "guidance_scale": 7.5,
    "prompt": args.prompt,
    "negative_prompt": args.negative_prompt,
}

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup pipelines
# inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     use_safetensors=True,
# ).to("cuda")

# refine_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=inpaint_pipe.text_encoder_2,
#     vae=inpaint_pipe.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# ).to("cuda")

inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
).to("cuda")

# Execute
diffusion_gen = InpaintingGenerativeV2(inpaint_pipe, hyper_params, device)

image = Image.open(args.input_path)
mask = Image.open(args.mask_path)

# Generate Image
output_Image = diffusion_gen.forward(image=image, mask=mask)
output_Image.save("./output/output.png")
