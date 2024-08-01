import torch
import os
from .processor import Processor
from scipy.optimize import minimize
from PIL import Image
from tqdm import tqdm
import numpy as np


def fast_dng_to_jpg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dng_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".dng"):
            dng_files.append(os.path.join(input_dir, filename))

    for f in tqdm(sorted(dng_files)):
        out_file = os.path.join(output_dir, os.path.split(f)[-1])[:-4] + ".jpg"
        if os.path.isfile(out_file):
            print(out_file, "skip", flush=True)
            continue
        else:
            print(out_file, "creating", flush=True)

        pp = Processor(f)

        # minimize(pp, np.array([0.1, -0.1, 0.1]), options=dict(eps=1e-3, maxiter=2))

        for i in range(6):
            pp(torch.randint(0, 3, (3,)).double().sub(1).div(10).numpy())

        for i in np.arange(0.1, 0.007, -0.005):
            pp(pp.best_param + torch.randn(3).double().mul(i * 3 / 5).numpy())

        img = pp.process_best()

        img = img.squeeze(0).permute(1, 2, 0)
        img = img.mul(255).round().clamp(0, 255).byte().cpu().numpy()
        Image.fromarray(img).save(out_file, quality=85)