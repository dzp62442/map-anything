import json

import numpy as np
import PIL
import torch
from PIL import Image


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def load_info(info):
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    # opencv cam -> opengl cam, maybe not necessary!
    # flip_yz = np.eye(4)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    # c2w = c2w@flip_yz

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return img_path, c2w, w2c


def load_conditions(img_paths, reso, is_input=False, load_rel_depth=False):
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, PIL.Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            # img.resize((w, h))
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = (
                fx * scale_w,
                fy * scale_h,
                cx * scale_w,
                cy * scale_h,
            )
            ck = np.array(
                [[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]]
            )
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag

    imgs, cks = [], []
    depths = []
    depths_m = []
    confs_m = []
    masks = []
    for img_path in img_paths:
        # param
        param_path = img_path.replace(
            "samples", "samples_param_small"
        )  # 224x400 resolution
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"])

        # img
        img_path = img_path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path)
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)

        # relative depth from DepthAnything-v2
        if load_rel_depth:
            depth_path = img_path.replace("sweeps_small", "sweeps_dpt_small")
            depth_path = depth_path.replace("samples_small", "samples_dpt_small")
            depth_path = depth_path.replace(".jpg", ".npy")
            disp = np.load(depth_path).astype(np.float32)
            if resize_flag:
                disp = Image.fromarray(disp)
                disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
                disp = np.array(disp)
            # inverse disparity to relative depth
            # clamping the farthest depth to 50x of the nearest
            range = np.minimum(disp.max() / (disp.min() + 0.001), 50.0)
            max = disp.max()
            min = max / range
            depth = 1 / np.maximum(disp, min)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depths.append(depth)
        else:
            depths.append(False)

        # metric depth from Metric3D-v2
        depthm_path = img_path.replace("sweeps_small", "sweeps_dptm_small")
        depthm_path = depthm_path.replace("samples_small", "samples_dptm_small")
        depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
        conf_path = depthm_path.replace("_dpt.npy", "_conf.npy")
        dptm = np.load(depthm_path).astype(np.float32)
        conf = np.load(conf_path).astype(np.float32)
        if resize_flag:
            dptm = Image.fromarray(dptm)
            dptm = dptm.resize((reso[1], reso[0]), Image.BILINEAR)
            dptm = np.array(dptm)
            conf = Image.fromarray(conf)
            conf = conf.resize((reso[1], reso[0]), Image.BILINEAR)
            conf = np.array(conf)
        depths_m.append(dptm)
        confs_m.append(conf)

        # 动态物体掩码
        if is_input:  # 输入图像使用全白掩码
            mask = np.ones(tuple(reso), dtype=np.float32)
        else:  # 输出图像读取真实掩码
            mask_path = img_path.replace("sweeps_small", "sweeps_mask_small")
            mask_path = mask_path.replace("samples_small", "samples_mask_small")
            mask_path = mask_path.replace(".jpg", ".png")
            mask = Image.open(mask_path).convert("L")  # convert to grayscale
            if resize_flag:
                mask = mask.resize((reso[1], reso[0]), Image.BILINEAR)
            mask = np.array(mask).astype(np.float32)
            mask = mask / 255.0
        masks.append(mask)

    imgs = torch.from_numpy(np.stack(imgs, axis=0)).float() / 255.0  # [v c h w]
    depths = torch.from_numpy(np.stack(depths, axis=0)).float()  # [v h w]
    depths_m = torch.from_numpy(np.stack(depths_m, axis=0)).float()  # [v h w]
    confs_m = torch.from_numpy(np.stack(confs_m, axis=0)).float()  # [v h w]
    masks = torch.from_numpy(np.stack(masks, axis=0)).float()  # [v h w]
    cks = torch.as_tensor(cks, dtype=torch.float32)

    return imgs, depths, depths_m, confs_m, masks, cks
