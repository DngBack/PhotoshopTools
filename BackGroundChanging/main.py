import os
import pprint
import random
import warnings
import torch
import numpy as np
from inference import Inference
from diffusers import StableDiffusionInpaintPipeline
import PIL.Image as Image
from pathlib import Path
import cv2
from RealESRGAN import RealESRGAN
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from resize import resize_and_pad, recover_size

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

    # Load pretrain
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Set Path
    img_url = "./data/custom_dataset/freestock_105548927.jpg"
    mask_url = "./mask/custom_dataset/freestock_105548927.png"

    # # Get path image
    # img = cv2.imread("./data/custom_dataset/freestock_105548927.jpg")
    # mask = cv2.imread(
    #     "./mask/custom_dataset/freestock_105548927.png", cv2.IMREAD_GRAYSCALE
    # )

    # resize with pad
    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    # Get some config
    prompt = "Warm Coffee House with some plants"
    device = "cuda"
    generator = torch.Generator(device="cuda").manual_seed(0)
    # guidance_scale = 7.5
    step = 100
    # height, width, _ = img.shape

    image = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        guidance_scale=8.0,
        num_inference_steps=20,  # steps between 15 and 30 work well for us
        strength=0.99,  # make sure to use `strength` below 1.0
        generator=generator,
    ).images[0]

    # Use API to converse
    # img_change_back = pipe(
    #     prompt=prompt,
    #     image=Image.fromarray(img_padded),
    #     mask_image=Image.fromarray(255 - mask_padded),
    #     num_inference_steps=step,
    # ).images[0]

    # Recover the size
    # img_resized, mask_resized = recover_size(
    #     np.array(img_change_back), mask_padded, (height, width), padding_factors
    # )

    # img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Set up up solution
    # model = RealESRGAN(device, scale=4)
    # model.load_weights("weights/RealESRGAN_x4.pth", download=True)

    # # resolution image
    # sr_image = model.predict(img_resized)
    # sr_image = sr_image.resize((width, height))

    image.save("./output/output.png")


if __name__ == "__main__":
    main(args)
