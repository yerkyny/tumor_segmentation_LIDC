import monai.transforms as M
import numpy as np

def pre_transforms(image, mask):
    
    noise = np.random.normal(loc=0, scale=20, size=image.shape)
    noisy_slice = image + noise
    noisy_slice = np.clip(noisy_slice, -1024, 300)
    return noisy_slice, mask


def augmentations():
    result = M.Compose([])
    return result


def post_transforms(image, mask):
    return image, mask
