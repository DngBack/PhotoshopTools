import warnings
import torch
import numpy as np
from inference.inference import Inference
import PIL.Image as Image
import cv2
from RealESRGAN import RealESRGAN
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import pipe


def resolution(image, height, width, device):
    model = RealESRGAN(device, scale=4)
    model.load_weights("weights/RealESRGAN_x4.pth", download=True)

    # resolution image
    sr_image = model.predict(image)
    sr_image = sr_image.resize((width, height))

    return sr_image
