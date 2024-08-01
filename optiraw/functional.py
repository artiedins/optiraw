import torch
import numpy as np
import rawpy
import math
import kornia


def read_dng(dng_file):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with rawpy.imread(dng_file) as raw:
        img = torch.from_numpy(raw.raw_image_visible.astype(np.float32))
        img = torch.stack([img[::2, ::2], (img[1::2, ::2] + img[::2, 1::2]) / 2, img[1::2, 1::2]])[None]
        img = img.to(device)

        low = (raw.black_level_per_channel[1] + raw.black_level_per_channel[3]) / 2
        if raw.camera_white_level_per_channel is None:
            high = float(raw.white_level)
        else:
            high = (raw.camera_white_level_per_channel[1] + raw.camera_white_level_per_channel[3]) / 2

        img.sub_(low).div_(high - low)
        flip = int(raw.sizes.flip)
        wb = np.array(raw.camera_whitebalance, dtype=np.float32)
        color_mat = raw.color_matrix.astype(np.float32)
        wb = torch.from_numpy(wb)[:3]
        color_mat = torch.from_numpy(color_mat)[:3, :3]

    return dict(img=img, flip=flip, wb=wb.to(device), color_mat=color_mat.to(device))


def process_image(d):
    img = d["img"].clone()
    wb = d["wb"].clone()
    flip = d["flip"] + 1
    param = d["param"][:]

    if flip == 1:
        pass
    elif flip == 2:
        # Flipped horizontally
        img = torch.flip(img, [3])
    elif flip == 3:
        # Rotated 180 degrees
        img = torch.rot90(img, k=2, dims=[2, 3])
    elif flip == 4:
        # Flipped vertically
        img = torch.flip(img, [2])
    elif flip == 5:
        # Rotated 90 degrees clockwise, then flipped horizontally
        img = torch.rot90(img, k=1, dims=[2, 3])
        img = torch.flip(img, [3])
    elif flip == 6:
        # Rotated 90 degrees clockwise
        img = torch.rot90(img, k=1, dims=[2, 3])  # 1 or -1
    elif flip == 7:
        # Rotated 90 degrees clockwise, then flipped vertically
        img = torch.rot90(img, k=-1, dims=[2, 3])
        # img = torch.flip(img, [2])
        pass
    elif flip == 8:
        # Rotated 270 degrees clockwise (or 90 degrees counter-clockwise)
        img = torch.rot90(img, k=3, dims=[2, 3])  # 3 or -3

    wb[0] = wb[0] * math.exp(param[0])
    wb[2] = wb[2] * math.exp(param[1])

    max_green = 1 - 10 ** (-3)
    over_exposed = img[:, 1] > max_green
    for i in range(3):
        img[:, i][over_exposed] = max_green / wb[i]

    wb_color_mat = d["color_mat"] * wb.view(1, 3)
    img = torch.einsum("bjhw,ij->bihw", img, wb_color_mat)

    mask = img <= 0.0031308
    img[mask] = 12.92 * img[mask]
    img[~mask] = 1.055 * torch.pow(img[~mask], 1 / 2.4) - 0.055

    mult = (2 - img[:, 1]).clamp_min(1).pow(param[2] * 10)
    for i in range(3):
        img[:, i] = img[:, i] * mult

    return img


def f1(x):
    return ((((-0.52525557 * x + 1.28619394) * x + 0.26472490) * x - 2.02598294) * x + 0.01301884) * x + 0.9985746516646661


def f2(x):
    return (((((0.02984797 * x - 0.49421622) * x + 3.37290890) * x - 12.01173602) * x + 23.27879108) * x - 22.97934763) * x + 8.875793499799272


def dists_to_weights_inplace(x):
    p1 = 1.29575
    p2 = 2.95609846
    is_f1 = x < p1
    is_f2 = np.logical_and(x < p2, ~is_f1)
    is_f3 = np.logical_and(~is_f1, ~is_f2)
    x[is_f1] = f1(x[is_f1])
    x[is_f2] = f2(x[is_f2])
    x[is_f3] = 0


def filter_inds_weights(ow, nw):
    rw = nw / ow
    ind_size = int(np.ceil(6 / rw))
    inds = np.zeros((nw, ind_size), dtype=np.int64)
    dists = np.zeros((nw, ind_size), dtype=np.float64) + 10

    olist = np.arange(ow) * rw + 0.5 * (rw - 1)
    for ni in range(nw):
        row_dists = np.abs(olist - ni)
        ind = np.where(row_dists < 3)[0]
        inds[ni, : len(ind)] = ind
        dists[ni, : len(ind)] = row_dists[ind]

    dists_to_weights_inplace(dists)
    inds = torch.from_numpy(inds)
    dists = torch.from_numpy(dists.astype(np.float32))

    wsum = torch.abs(dists.sum(1, keepdim=True))
    wsum.clamp_(min=1e-7)
    dists /= wsum

    return inds, dists


def resize(img, long_side):
    img = img.clamp(0, 1)
    img = kornia.geometry.transform.resize(img, long_side, side="long", interpolation="bicubic")
    return img

    bs, ch, oh, ow = img.shape

    if oh > ow:
        nh = int(long_side)
        nw = int(round(nh * ow / oh))
    else:
        nw = int(long_side)
        nh = int(round(nw * oh / ow))

    img = img.contiguous().view(bs * ch * oh, ow)
    inds, weights = filter_inds_weights(ow, nw)  # both nw, nsamples; ind are into rows of ow
    inds = inds.to(img.device)
    weights = weights.to(img.device)

    samples = inds.shape[1]
    img = img.t()  # ow, bs
    img = (img[inds.view(-1)].view(nw, samples, bs * ch * oh) * weights.view(nw, samples, 1)).sum(1)
    img = img.t()
    img = img.view(bs, ch, oh, nw)

    img = img.permute(0, 1, 3, 2).contiguous()  # nw, oh

    img = img.view(bs * ch * nw, oh)
    inds, weights = filter_inds_weights(oh, nh)
    inds = inds.to(img.device)
    weights = weights.to(img.device)

    samples = inds.shape[1]
    img = img.t()  # oh, bs
    img = (img[inds.view(-1)].view(nh, samples, bs * ch * nw) * weights.view(nh, samples, 1)).sum(1)
    img = img.t()
    img = img.view(bs, ch, nw, nh)

    img = img.permute(0, 1, 3, 2)

    return img
