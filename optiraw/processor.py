import torch
import numpy as np
from .functional import process_image, read_dng, resize

# from pyiqa.utils.registry import ARCH_REGISTRY
import pyiqa


class Processor:
    def __init__(self, dng_file):
        self.d = read_dng(dng_file)

        self.best_param = np.array([0.0, 0.0, 0.05])
        self.best_loss = float("inf")

        # self.iqa_model = ARCH_REGISTRY.get("CLIPIQA")(model_type="clipiqa+")
        # self.iqa_model.to(self.d["img"].device)
        # self.iqa_model.eval()

        self.iqa_model = pyiqa.create_metric("clipiqa+", metric_mode="NR", device=self.d["img"].device)
        # self.iqa_model = pyiqa.create_metric("musiq", metric_mode="NR", device=self.d["img"].device)

    def __call__(self, param):
        self.d["param"] = param.tolist()
        img = process_image(self.d)
        img = resize(img, 640)

        with torch.inference_mode():
            loss = 1 / self.iqa_model(img).view(-1).item()

        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_param = param.copy()

        return loss

    def process_best(self):
        self.d["param"] = self.best_param.tolist()
        img = process_image(self.d, print_param=True)
        return img
