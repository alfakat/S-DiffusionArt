"""
This script estimate depth map on input image, save it as image and return as tensor,
so can be further used in processing.
"""

import torch
import numpy as np
from PIL import Image
from datetime import datetime

from transformers import pipeline
from diffusers.utils import load_image

depth_estimator = pipeline("depth-estimation")
image_path = load_image('red_apples.png')


def estimate_depth_map(image, depth_estimator):
    image = np.array(depth_estimator(image)["depth"])[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    depth_image = Image.fromarray(image)
    depth_image.save(f"{datetime.now().strftime('%d%m%Y_%H%M%S')}.png")
    return depth_map


estimate_depth_map(image_path, depth_estimator).unsqueeze(0).half().to("cuda")
