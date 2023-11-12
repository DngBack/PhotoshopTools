import torch

import numpy as np

import cv2
from PIL import Image


class DiffusionGeneration:
    """
    Stable Diffusion for generation process.
    """

    def __init__(self, inpaint_pipe, refine_pipe, hp_dict, device):
        """
        Args:
            inpaint_pipe : Stable Diffusion inpaint pipeline. Note that: input size = 512
            refine_pipe : Stable Diffusion refiner pipeline. Note that: input size = 1024
            hp_dict (dict): Hyperparameters dicitionary for generation.
            device (torch.device): Device used.
        """
        # Setup device
        self.device = device

        # Setup pipelines
        self.inpaint_pipe = inpaint_pipe.to(self.device)
        self.refine_pipe = refine_pipe.to(self.device)

        # Setup hyperparameters dictionary
        self.hp_dict = hp_dict

    def inpaint_generate_image(self, image, mask):
        """
        Inpainting function

        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Mask image

        Returns:
            inpainted_image (PIL.Image): Inpainted image
        """
        # Save ori_size of input image to reconstruct
        ori_size = image.size

        # Resize image and mask to passing model
        input_image = image.resize((512, 512))
        input_mask = mask.resize((512, 512))

        # Apply pipeline
        generator = torch.Generator(self.device).manual_seed(self.hp_dict["seed"])
        result = self.inpaint_pipe(
            image=input_image,
            mask=input_mask,
            prompt=self.hp_dict["prompt"],
            negative_prompt=self.hp_dict["negative_prompt"],
            generator=generator,
        )
        output_image = result.images[0]

        # Resize inpainted image to original size
        inpainted_image = output_image.resize(ori_size)

        return inpainted_image

    def refiner_generate_image(self, image, mask):
        """
        Refiner function

        Args:
            image (PIL.Image): Input image with
            mask (PIL.Image): Dilated mask image

        Returns:
            refined_image (PIL.Image): Refined image
        """
        # Save ori_size of input image to reconstruct
        ori_size = image.size

        # Apply pipeline
        result = self.refine_pipe(
            prompt=self.hp_dict["prompt"],
            image=image,
            mask_image=mask,
            # guidance_scale = hp_dict["guidance_scale"],
            num_inference_steps=self.hp_dict["num_inference_steps"],
            denoising_start=self.hp_dict["denoising_start"],
        )
        output_image = result.images[0]

        # Resize refined image to original size
        refined_image = output_image.resize(ori_size)

        return refined_image

    def dilate_mask(self, init_mask):
        """
        Make dilated mask from input mask

        Args:
            init_mask (np.array): Input mask

        Returns:
            dilated_mask_pil (PIL.Image): Dilated mask as PIL Image
        """
        kernel = np.ones(self.hp_dict["kernel_size"], np.uint8)
        img_dilation = cv2.dilate(
            init_mask, kernel, iterations=self.hp_dict["kernel_iterations"]
        )
        dilated_mask = Image.fromarray(img_dilation)
        return dilated_mask

    def forward(self, image, mask, is_dilated=False):
        """
        Generation process

        Args:
            image (PIL.Image): Input image
            mask (PIL.Image): Mask image
            is_dilated (bool): Check if generate dilated mask

        Returns:
            final_image (PIL.Image): Final result image
        """
        # Inpaiting process
        inpainted_image = self.inpaint_generate_image(image, mask)
        
        # Get mask image
        if is_dilated:
            mask = self.dilate_mask(mask)
        
        # Refining process
        refined_image = self.refiner_generate_image(inpainted_image, mask)
        return refined_image




