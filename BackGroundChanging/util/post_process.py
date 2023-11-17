import cv2
import numpy as np
from PIL import Image


class PostProcessing:
    def __init__(self, ori_image, mask, diff_image):
        self.ori_image = ori_image
        self.mask = mask
        self.diff_image = diff_image

    def get_transparent_object(self):
        trans_image = self.ori_image.copy()
        trans_image.putalpha(self.mask)
        return trans_image

    def overlay_object2output(self):
        trans_image = self.get_transparent_object()
        trans_image = trans_image.convert("RGBA")
        self.diff_image.paste(trans_image, trans_image)
        return self.diff_image


def make_transparent_mask(img_ori: np.array, img_mask: np.array):
    """
    Using contour filtering to determine target object, after that extract a transparent image
    from original image.

    Args:
        img_ori (np.array): Original image (H, W, 3)
        img_mask (np.array): Output mask image of Tracer model. (H, W, 3)

    Returns:
        transparent_mask (np.array): Transparent image after filtering.
    """
    # Read image mask as a Grayscale image
    mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)

    # Finding contour of mask
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtered = cv2.drawContours(mask, contour, -1, (0, 0, 255), 2)

    # Extract from original image to isolated image with filtered mask
    isolated_img = cv2.bitwise_and(img_ori, img_ori, mask=mask_filtered)

    # Convert isolated image to transparent image
    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(isolated_img)
    rgba = [b, g, r, alpha]
    transparent_mask = cv2.merge(rgba, 4)

    return transparent_mask


def replace_object(img: np.array, transparent_img: np.array):
    """
    Replace object of img by transparent image

    Args:
        img (np.array): Image what want to change object (H, W, 3)
        transparent_img (np.array): Target object what is pasted on input image (H, W, 4)

    Returns:
        replace_img (np.array): Output image (H, W, 3)
    """
    # Extract the alpha channel from the foreground
    alpha_channel = transparent_img[:, :, 3] / 255.0

    # Create a mask using alpha channel
    mask = np.stack([alpha_channel] * 3, axis=2)

    # Blend the foreground and background using alpha blending
    replace_img = (1.0 - mask) * img + mask * transparent_img[:, :, :3]
    return replace_img
