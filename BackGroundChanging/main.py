import os
import random
import warnings
import torch
import numpy as np
from inference.inference import Inference
import PIL.Image as Image
import cv2
import time
from RealESRGAN import RealESRGAN
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from model.resolution import resolution
from model.bgChanging import ChangingBg
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import DiffusionPipeline
from model.diffusion_gen import *

from config import getConfig

warnings.filterwarnings("ignore")
args = getConfig()


def main(args):
    # Random Seed
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Some Config Path
    img_url = "./data/custom_dataset/Image.png"
    mask_url = "./mask/custom_dataset/Image.png"
    maskReplace_url = "./mask_replace/mask_replace.png"
    output_url = "./output/output.png"

    # Get image
    input_url = "./Image/Test1.jpg"
    inputImage = cv2.imread(input_url)
    save_input = cv2.imwrite(img_url, inputImage)

    # Remove Back ground and get
    save_path = os.path.join(
        args.model_path, args.dataset, f"TE{args.arch}_{str(args.exp_num)}"
    )

    # Get pre-mask
    t_getmask = time.time()
    Inference(args, save_path).test()
    print("Time of get mask processing: ", time.time() - t_getmask)

    # Get mask
    mask_image = cv2.imread(mask_url, cv2.IMREAD_GRAYSCALE)
    mask_image = 255 - mask_image
    cv2.imwrite(maskReplace_url, mask_image)

    # resize Image for stable diffusion
    image = load_image(img_url)
    mask_image = load_image(maskReplace_url)

    # Setup hyper parameters
    hp_dict = {
        "seed": -305,
        "kernel_size": (5, 5),
        "kernel_iterations": 15,
        "num_inference_steps": 70,
        "denoising_start": 0.70,
        "guidance_scale": 7.5,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Pipeline calling
    inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    refine_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=inpaint_pipe.text_encoder_2,
        vae=inpaint_pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    # Execute
    diffusion_gen = DiffusionGeneration_v2(inpaint_pipe, refine_pipe, hp_dict, device)

    # Get input
    image = Image.open("./output/output.png")
    mask = Image.open("./mask_replace/mask_replace.png")

    # Generate Image
    output_Image = diffusion_gen.forward(image, mask)
    output_Image.save(output_url)


if __name__ == "__main__":
    main(args)
