import importlib
import nibabel as nb
from torch.utils.data import Dataset
import numpy as np

class TumorDataset(Dataset):
    def __init__(self, items, config, stage):
        self.items = items
        self.config = config
        self.stage = stage

        self.module = importlib.import_module(config["TRANSFORMS"]["PY"])
        self.pre_transforms = getattr(self.module, config["TRANSFORMS"]["PRE_TRANS"])
        self.augmentations = getattr(
            self.module, config["TRANSFORMS"]["AUGMENTATIONS"]
        )()
        self.post_transforms = getattr(self.module, config["TRANSFORMS"]["POST_TRANS"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # ---- Image ----
        ct_path = self.items[idx]["cts_path"]
        ct = nb.load(ct_path).get_fdata()

        # ---- Label ----
        mask_path = self.items[idx]["masks_path"]
        mask = nb.load(mask_path).get_fdata()
        
        # ---- Pre transforms and Normalization -----
        ct, mask = self.pre_transforms(ct, mask)
        ct = (ct - ct.min()) / (ct.max() - ct.min())
        ct = ct.astype(np.float32)
        ct = np.pad(ct, ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(mask, ((0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)

        if self.stage == "train":
            ct = self.augmentations(ct)
            mask = self.augmentations(mask)

        # ---- Post Transforms ----
        ct, mask = self.post_transforms(ct, mask)
        ct = ct.astype(np.float32)
        
        ct = np.expand_dims(ct, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return ct, mask

