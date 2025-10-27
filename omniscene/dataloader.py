import os
import os.path as osp

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import copy
import json
import pickle as pkl

import cv2
import torch
from torch.utils.data import Dataset
from transforms.loading import load_conditions, load_info

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

bins_demo = [
    "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
    "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
    "scene04219bfdc9004ba2af16d3079ecc4353_bin061",
    "scene07aed9dae37340a997535ad99138e243_bin058",
    "scene0ac05652a4c44374998be876ba5cd6fd_bin121",
    "scene16e50a63b809463099cb4c378fe0641e_bin231",
    "scene197a7e4d3de84e57af17b3d65fcb3893_bin177",
    "scene19d97841d6f64eba9f6eb9b6e8c257dc_bin001",
    "scene201b7c65a61f4bc1a2333ea90ba9a932_bin071",
    "scene2086743226764f268fe8d4b0b7c19590_bin043",
    "scene265f002f02d447ad9074813292eef75e_bin128",
    "scene26a6b03c8e2f4e6692f174a7074e54ff_bin103",
    "scene2abb3f3517c64446a5768df5665da49d_bin128",
    "scene2ca15f59d656489a8b1a0be4d9bead4e_bin003",
    "scene2ed0fcbfc214478ca3b3ce013e7723ba_bin154",
    "scene2f56eb47c64f43df8902d9f88aa8a019_bin136",
    "scene3045ed93c2534ec2a5cabea89b186bd9_bin176",
    "scene36f27b26ef4c423c9b79ac984dc33bae_bin207",
    "scene3a2d9bf6115f40898005d1c1df2b7282_bin107",
    "scene3ada261efee347cba2e7557794f1aec8_bin005",
    "scene3dd2be428534403ba150a0b60abc6a0a_bin083",
    "scene3dd9ad3f963e4f588d75c112cbf07f56_bin131",
    "scene3f90afe9f7dc49399347ae1626502aed_bin095",
    "scene4962cb207a824e57bd10a2af49354b16_bin089",
    "scene5301151d8b6a42b0b252e95634bd3995_bin121",
    "scene5521cd85ed0e441f8d23938ed09099dd_bin067",
    "scene6a24a80e2ea3493c81f5fdb9fe78c28a_bin033",
    "scene6af9b75e439e4811ad3b04dc2220657a_bin115",
    "scene7061c08f7eec4495979a0cf68ab6bb79_bin180",
    "scene7365495b74464629813b41eacdb711af_bin067",
    "scene76ceedbcc6a54b158eba9945a160a7bc_bin063",
    "scene7e8ff24069ff4023ac699669b2c920de_bin014",
    "scene813213458a214a39a1d1fc77fa52fa34_bin040",
    "scene848ac962547c4508b8f3b0fcc8d53270_bin023",
    "scene85651af9c04945c3a394cf845cb480a6_bin017",
    "scene8edbc31083ab4fb187626e5b3c0411f7_bin017",
    "scene9088db17416043e5880a53178bfa461c_bin005",
    "scene91c071bcc1ad4fa1b555399e1cfbab79_bin002",
    "scene91f797db8fb34ae5b32ba85eecae47c9_bin004",
    "scene9709626638f5406f9f773348150d02fd_bin092",
    "sceneafbc2583cc324938b2e8931d42c83e6b_bin009",
    "sceneb07358651c604e2d83da7c4d4755de73_bin017",
    "sceneb94fbf78579f4ff5ab5dbd897d5e3199_bin155",
    "scenecba3ddd5c3664a43b6a08e586e094900_bin032",
    "scened3b86ca0a17840109e9e049b3dd40037_bin040",
    "scenee036014a715945aa965f4ec24e8639c9_bin005",
    "sceneefa5c96f05594f41a2498eb9f2e7ad99_bin092",
    "scenef97bf749746c4c3a8ad9f1c11eab6444_bin009",
]

bins_dynamic_demo = [
    "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
    "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
    "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
    "scene84e056bd8e994362a37cba45c0f75558_bin100",
    "scene717053dec2ef4baa913ba1e24c09edff_bin000",
    "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
    "scene724957e51f464a9aa64a16458443786d_bin000",
    "scened3c39710e9da42f48b605824ce2a1927_bin050",
    "scene034256c9639044f98da7562ef3de3646_bin000",
    "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
    "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
    "scene50ff554b3ecb4d208849d042b7643715_bin000",
]


class nuScenesDataset(Dataset):
    data_root: str = "/home/B_UserData/dongzhipeng/Datasets/dataset_omniscene"
    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(self, resolution: list = [224, 400], split: str = "train"):
        super().__init__()

        self.reso = resolution

        # load bin tokens
        if split == "train":
            # for training
            self.bin_tokens = json.load(
                open(
                    osp.join(self.data_root, self.data_version, "bins_train_3.2m.json")
                )
            )["bins"]
        elif split == "val":
            # for visualization during training
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[:30000:3000][
                :10
            ]  # 前 30000 个 bin tokens 中每隔 3000 个取一个，取 10 个
        elif split == "test":
            # for evaluation
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json"))
            )["bins"]
        elif split == "mini-test":
            # for evaluation mini test
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[0::14][
                :2048
            ]  # 每隔 14 个取一个，取 2048 个
        elif split == "demo":
            # super mini test
            self.bin_tokens = bins_dynamic_demo

        self.split = split

    def __len__(self):
        return len(self.bin_tokens)

    def __getitem__(self, index):
        bin_token = self.bin_tokens[index]
        with open(
            osp.join(
                self.data_root, self.data_version, "bin_infos_3.2m", bin_token + ".pkl"
            ),
            "rb",
        ) as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        # =================== Input views of this bin ===================== #
        input_img_paths, input_c2ws = [], []
        for cam in self.camera_types:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        # load and modify images (cropped or resized if necessary), and modify intrinsics accordingly
        (
            input_imgs,
            input_depths,
            input_depths_m,
            input_confs_m,
            input_masks,
            input_cks,
        ) = load_conditions(
            input_img_paths, self.reso, is_input=True, load_rel_depth=False
        )
        input_cks = torch.as_tensor(input_cks, dtype=torch.float32)

        batch_dict = {
            "bin_token": bin_token,
            "img_paths": input_img_paths,
            "imgs": input_imgs,
            "intrinsics": input_cks,
            "poses": input_c2ws,
        }

        return batch_dict
