import os
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from model.TRACER import TRACER
from torch.utils import model_zoo
from config import getConfig
import numpy as np
import cv2


class Background_Extraction:
    """
    Background Extraction

    Args:
        model (nn.Module): The model to extract background. Note that: This model is trained parallel, if you want to infer the model please set strict=False)
        device (torch.device): The device to extract background
        pretrained_url (str): The url of pretrained model
        save_path (str): The path to save the result
        img_size (int): The default size of model

    Returns:
        (Map, Edge, (DS_map) : Map image, Edge image, DS_map

    """

    def __init__(
        self,
        model,
        device,
        pretrained_url: str = None,
        save_path: str = None,
        img_size=640,
    ):
        self.img_size = img_size
        self.device = device

        self.pretrained_url = pretrained_url
        self.save_path = save_path

        self.model = model.to(self.device)

        # Setting transformation for test image
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load model from url
        state_sict = model_zoo.load_url(self.pretrained_url, map_location=self.device)
        self.model.load_state_dict(state_sict, strict=False)

    def extract(self, img):
        print(img.size)
        ori_size = (img.size[0], img.size[1])
        self.model.eval()

        with torch.no_grad():
            img = self.transform(img).unsqueeze(0).to(self.device)
            # print(img.shape)
            output = self.model(img)
            output = F.interpolate(
                output[0].unsqueeze(0), size=ori_size, mode="bilinear"
            )
            print(output.shape)
            # print(output.shape)
            # transforms.Grayscale(num_output_channels=1),

            # img_transform = transforms.ToPILImage()

            # output = img_transform(output.squeeze())
            # output.show()
            # os.makedirs(os.path.join(self.save_path), exist_ok=True)
            # output.save(os.path.join(self.save_path, "result.png"))
            # return output
            output = (output.squeeze().detach().cpu().numpy() * 255.0).astype(np.uint8)
            print(output.shape)
            # cv2.imwrite("t.jpg", output)
            cv2.imshow("Test", output)
            cv2.waitKey(0)


if __name__ == "__main__":
    args = getConfig()
    device = torch.device("cpu")
    model = TRACER(args)

    pretrained_url = "https://github.com/Karel911/TRACER/releases/download/v1.0/TRACER-Efficient-7.pth"
    save_path = "./out_bg"

    bg_extract = Background_Extraction(model, device, pretrained_url, save_path)

    imgs = Image.open("./Image/Test1.jpg").convert("RGB")

    out = bg_extract.extract(imgs)
