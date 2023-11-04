import os
import pprint
import random
import warnings
import torch
import numpy as np
from BackGroundChanging.inference.inference import Inference
import PIL.Image as Image
from pathlib import Path
import cv2
from RealESRGAN import RealESRGAN
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import pipe

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

    save_path = os.path.join(
        args.model_path, args.dataset, f"TE{args.arch}_{str(args.exp_num)}"
    )

    Inference(args, save_path).test()

    # Load pretrain
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Set Path
    img_url = "./data/custom_dataset/fd56c4e09c.jpg"
    mask_url = "mask_replace.png"

    # Get mask
    mask_image = cv2.imread(
        "./mask/custom_dataset/fd56c4e09c.png", cv2.IMREAD_GRAYSCALE
    )

    height, width = mask_image.shape

    mask_image = 255 - mask_image

    filename = "mask_replace.png"
    status = cv2.imwrite(filename, mask_image)

    # resize with pad
    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    # Get some config
    prompt = "Office in Maketing Company "
    device = "cuda"
    generator = torch.Generator(device="cuda").manual_seed(0)

    image_out = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]

    # Resized Image
    img_resized = image_out.resize((height, width))

    # Set up up solution
    model = RealESRGAN(device, scale=4)
    model.load_weights("weights/RealESRGAN_x4.pth", download=True)

    # resolution image
    sr_image = model.predict(img_resized)
    sr_image = sr_image.resize((width, height))

    sr_image.save("./output/output.png")


if __name__ == "__main__":
    main(args)
