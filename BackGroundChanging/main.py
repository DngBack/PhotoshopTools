import os
import pprint
import random
import warnings
import torch
import numpy as np
from inference.inference import Inference
import PIL.Image as Image
from pathlib import Path
import cv2
from RealESRGAN import RealESRGAN
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import pipe
from resolution import resolution
from bgChanging import ChangingBg

from config import getConfig

warnings.filterwarnings("ignore")
args = getConfig()


def main(args):
    print("<---- Training Params ---->")
    pprint.pprint(args)

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
    input_url = "./Image/Test2.png"
    inputImage = cv2.imread(input_url)
    save_input = cv2.imwrite(img_url, inputImage)

    # Remove Back ground and get
    save_path = os.path.join(
        args.model_path, args.dataset, f"TE{args.arch}_{str(args.exp_num)}"
    )

    Inference(args, save_path).test()

    # Load pretrain
    # pipe = AutoPipelineForInpainting.from_pretrained(
    #     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    # ).to("cuda")

    # Get mask
    mask_image = cv2.imread(mask_url, cv2.IMREAD_GRAYSCALE)
    height, width = mask_image.shape
    mask_image = 255 - mask_image
    cv2.imwrite(maskReplace_url, mask_image)

    # resize Image for stable diffusion
    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(maskReplace_url).resize((1024, 1024))

    # Get some config for
    prompt = "A boy, coffe house, book, brown table, lamp, a television"
    device = "cuda"
    generator = torch.Generator(device="cuda").manual_seed(0)

    # image_out = pipe(
    #     prompt=prompt,
    #     image=image,
    #     mask_image=mask_image,
    #     guidance_scale=8.0,
    #     num_inference_steps=20,  # steps between 15 and 30 work well for us
    #     strength=0.99,  # make sure to use `strength` below 1.0
    #     generator=generator,
    # ).images[0]
    image_out = ChangingBg(image, mask_image, prompt, generator)

    # Resized Image
    img_resized = image_out.resize((height, width))

    sr_image = resolution(img_resized, height, width, device)

    sr_image.save(output_url)


if __name__ == "__main__":
    main(args)
