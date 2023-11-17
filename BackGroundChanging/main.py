import os
import random
import warnings
import torch
import numpy as np
from inference.inference import Inference
from PIL import Image, ImageOps
import cv2
import time
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from model.resolution import resolution
from model.bgChanging import ChangingBg
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers import DiffusionPipeline
from model.diffusion_gen import *
from util.post_process import *

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
    output_url = "./output/output.png"
    output_final_url = "./output_final/output_final.png"

    # Get image
    input_url = args.input_path
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
    diffusion_gen = DiffusionGeneration(inpaint_pipe, refine_pipe, hp_dict, device)

    # Get input
    image = Image.open(img_url)
    mask = Image.open(mask_url)

    # Generate Image
    output_Image = diffusion_gen.forward(image=image, mask=ImageOps.invert(mask))
    output_Image.save(output_url)

    # Post Processing
    # Get input
    ori_image = Image.open(img_url)
    mask = Image.open(mask_url)
    diff_image = Image.open(output_url)
    # Execute
    post_processing = PostProcessing(ori_image, mask, diff_image)
    output_final = post_processing.overlay_object2output()
    output_final.save(output_final_url)


if __name__ == "__main__":
    main(args)
