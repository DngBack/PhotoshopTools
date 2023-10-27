import os
import pprint
import random
import warnings
import torch
import numpy as np
from trainer import Trainer, Tester
from inference import Inference
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
from pathlib import Path
import cv2
from RealESRGAN import RealESRGAN
from resize import (
    resize_and_pad,
    recover_size,
    crop_for_filling_pre,
    crop_for_filling_post,
)

from config import getConfig

warnings.filterwarnings("ignore")
args = getConfig()


def replace_img_with_sd(img, mask, text_prompt, step=50, device="cuda"):
    guidance_scale = 1.5
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
    ).to(device)
    # img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)

    img_padded = pipe(
        prompt=text_prompt,
        image=Image.fromarray(img),
        mask_image=Image.fromarray(255 - mask),
        guidance_scale=guidance_scale,
        num_inference_steps=step,
    ).images[0]
    # mask_resized = np.expand_dims(mask_resized, -1) / 255
    # img_resized = img_resized * (1-mask_resized) + img * mask_resized
    img_resized = np.array(img_padded)
    return img_resized


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

    # if args.action == 'train':
    #     save_path = os.path.join(args.model_path, args.dataset, f'TE{args.arch}_{str(args.exp_num)}')

    #     # Create model directory
    #     os.makedirs(save_path, exist_ok=True)
    #     Trainer(args, save_path)

    # elif args.action == 'test':
    #     save_path = os.path.join(args.model_path, args.dataset, f'TE{args.arch}_{str(args.exp_num)}')
    #     datasets = ['DUTS', 'DUT-O', 'HKU-IS', 'ECSSD', 'PASCAL-S']

    #     for dataset in datasets:
    #         args.dataset = dataset
    #         test_loss, test_mae, test_maxf, test_avgf, test_s_m = Tester(args, save_path).test()

    #         print(f'Test Loss:{test_loss:.3f} | MAX_F:{test_maxf:.4f} '
    #               f'| AVG_F:{test_avgf:.4f} | MAE:{test_mae:.4f} | S_Measure:{test_s_m:.4f}')
    # else:

    save_path = os.path.join(
        args.model_path, args.dataset, f"TE{args.arch}_{str(args.exp_num)}"
    )

    Inference(args, save_path).test()

    # Get path image
    img = cv2.imread("./data/custom_dataset/freestock_105548927.jpg")
    mask = cv2.imread(
        "./mask/custom_dataset/freestock_105548927.png", cv2.IMREAD_GRAYSCALE
    )

    # resize with pad
    img_padded, mask_padded, padding_factors = resize_and_pad(img, mask)

    # Get some config
    prompt = "Warm Coffee House with some plants"
    device = "cuda"
    guidance_scale = 7.5
    step = 100
    height, width, _ = img.shape

    # Use API to converse
    img_change_back = replace_img_with_sd(
        img=img_padded,
        mask=mask_padded,
        text_prompt=prompt,
        guidance_scale=guidance_scale,
        step=step,
        device=device,
    )
    # cv2.imwrite("./output.png", img_background)

    # Recover the size
    img_resized, mask_resized = recover_size(
        np.array(img_padded), mask_padded, (height, width), padding_factors
    )

    # Set up up solution
    model = RealESRGAN(device, scale=4)
    model.load_weights("weights/RealESRGAN_x4.pth", download=True)

    # resolution image
    sr_image = model.predict(img_resized)

    sr_image.save("./output/output.png")


if __name__ == "__main__":
    main(args)
