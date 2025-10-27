# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Simplified MapAnything Gradio demo that loads a fixed multi-view scene (images + 内参 + 外参)，
直接执行推理并展示三维重建结果。
"""

import gc
import os
import time
from pathlib import Path
from typing import Dict, List

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gradio as gr
import numpy as np
import torch
from dataloader import nuScenesDataset

from mapanything.utils.geometry import depthmap_to_world_frame, points_to_normals
from mapanything.utils.hf_utils.css_and_html import (
    get_acknowledgements_html,
    get_description_html,
    get_gradio_theme,
    get_header_html,
    GRADIO_CSS,
    MEASURE_INSTRUCTIONS_HTML,
)
from mapanything.utils.hf_utils.hf_helpers import initialize_mapanything_model
from mapanything.utils.hf_utils.viz import predictions_to_glb
from mapanything.utils.image import preprocess_inputs, rgb

# -------------------------------------------------------------------------
# 固定数据路径（可替换为真实路径）
# -------------------------------------------------------------------------
OUTPUT_DIR = Path("./demo_omniscene_output")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_fixed_views() -> List[Dict]:
    omniscene_dataset = nuScenesDataset(resolution=[224, 400], split="demo")
    sample = omniscene_dataset[0]

    raw_views = []
    for idx in range(len(sample["img_paths"])):
        raw_views.append(
            {
                "img": sample["imgs"][idx],
                "intrinsics": sample["intrinsics"][idx],
                "camera_poses": sample["poses"][idx],
                "is_metric_scale": torch.ones(1, dtype=torch.bool),
                "idx": idx,
                "instance": str(idx),
            }
        )
    return raw_views


# MapAnything 配置
high_level_config = {
    "path": "configs/train.yaml",
    "hf_model_name": "facebook/map-anything",
    "model_str": "mapanything",
    "config_overrides": [
        "machine=aws",
        "model=mapanything",
        "model/task=images_only",
        "model.encoder.uses_torch_hub=false",
    ],
    "checkpoint_name": "model.safetensors",
    "config_name": "config.json",
    "trained_with_amp": True,
    "trained_with_amp_dtype": "bf16",
    "data_norm_type": "dinov2",
    "patch_size": 14,
    "resolution": 518,
}


# 全局缓存
model = None
PROCESSED_VIEWS: List[Dict] = []
PREDICTIONS: Dict[str, np.ndarray] = {}
PROCESSED_DATA: Dict[int, Dict] = {}
DEFAULT_GLB_PATH: str = ""


def get_logo_base64() -> str:
    import base64

    logo_path = Path("examples/WAI-Logo/wai_logo.png")
    if not logo_path.is_file():
        return ""
    with logo_path.open("rb") as img_file:
        return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"


def _ensure_pose_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        homo = np.eye(4, dtype=np.float32)
        homo[:3, :4] = pose
        return homo
    raise ValueError(f"camera pose shape must be (3,4) or (4,4), got {pose.shape}")


def preprocess_views(raw_views: List[Dict]) -> List[Dict]:
    return preprocess_inputs(
        raw_views,
        resize_mode="fixed_mapping",
        norm_type=high_level_config["data_norm_type"],
        patch_size=high_level_config["patch_size"],
        resolution_set=high_level_config["resolution"],
        verbose=False,
    )


def run_inference(views: List[Dict]) -> List[Dict]:
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = initialize_mapanything_model(high_level_config, device)
    else:
        model = model.to(device)
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        outputs = model.infer(
            views,
            apply_mask=True,
            mask_edges=True,
            memory_efficient_inference=False,
        )
    return outputs


def build_predictions(outputs: List[Dict]) -> Dict[str, np.ndarray]:
    extrinsics = []
    intrinsics = []
    world_points = []
    depth_maps = []
    images = []
    final_masks = []
    confidences = []

    for pred in outputs:
        depthmap = pred["depth_z"][0].squeeze(-1)
        intrinsics_t = pred["intrinsics"][0]
        camera_pose = pred["camera_poses"][0]
        conf = pred["conf"][0]

        pts3d_world, valid_mask = depthmap_to_world_frame(
            depthmap, intrinsics_t, camera_pose
        )

        if "mask" in pred:
            mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        else:
            mask = np.ones_like(depthmap.cpu().numpy(), dtype=bool)
        mask &= valid_mask.cpu().numpy()

        extrinsics.append(camera_pose.cpu().numpy())
        intrinsics.append(intrinsics_t.cpu().numpy())
        world_points.append(pts3d_world.cpu().numpy())
        depth_np = depthmap.cpu().numpy()
        depth_maps.append(depth_np[..., np.newaxis])
        images.append(pred["img_no_norm"][0].cpu().numpy())
        final_masks.append(mask)
        confidences.append(conf.cpu().numpy())

    predictions = {
        "extrinsic": np.stack(extrinsics, axis=0),
        "intrinsic": np.stack(intrinsics, axis=0),
        "world_points": np.stack(world_points, axis=0),
        "depth": np.stack(depth_maps, axis=0),
        "images": np.stack(images, axis=0),
        "final_mask": np.stack(final_masks, axis=0),
        "conf": np.stack(confidences, axis=0),
    }
    return predictions


def colorize_depth(depth_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if depth_map is None:
        return None
    depth_normalized = depth_map.copy().astype(np.float32)
    valid_mask = depth_normalized > 0
    if mask is not None:
        valid_mask = valid_mask & mask
    if valid_mask.sum() > 0:
        valid_depths = depth_normalized[valid_mask]
        p5 = np.percentile(valid_depths, 5)
        p95 = np.percentile(valid_depths, 95)
        denom = max(p95 - p5, 1e-6)
        depth_normalized[valid_mask] = (depth_normalized[valid_mask] - p5) / denom
    import matplotlib.pyplot as plt

    colored = plt.cm.turbo_r(depth_normalized)
    colored = (colored[:, :, :3] * 255).astype(np.uint8)
    colored[~valid_mask] = [255, 255, 255]
    return colored


def colorize_normal(normal_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    if normal_map is None:
        return None
    normal_vis = normal_map.copy()
    if mask is not None:
        invalid_mask = ~mask
        normal_vis[invalid_mask] = [0, 0, 0]
    normal_vis = (normal_vis + 1.0) / 2.0
    return (normal_vis * 255).astype(np.uint8)


def process_predictions_for_visualization(
    predictions: Dict[str, np.ndarray],
    views: List[Dict],
    filter_black_bg: bool = False,
    filter_white_bg: bool = False,
) -> Dict[int, Dict]:
    processed = {}
    for view_idx, view in enumerate(views):
        image = rgb(view["img"], norm_type=high_level_config["data_norm_type"])
        pred_pts3d = predictions["world_points"][view_idx]
        mask = predictions["final_mask"][view_idx].copy()

        if filter_black_bg:
            colors = image[0] * 255 if image[0].max() <= 1.0 else image[0]
            black_mask = colors.sum(axis=2) >= 16
            mask = mask & black_mask

        if filter_white_bg:
            colors = image[0] * 255 if image[0].max() <= 1.0 else image[0]
            white_mask = ~(
                (colors[:, :, 0] > 240)
                & (colors[:, :, 1] > 240)
                & (colors[:, :, 2] > 240)
            )
            mask = mask & white_mask

        normals, _ = points_to_normals(pred_pts3d, mask=mask)

        processed[view_idx] = {
            "image": (image[0] * 255).astype(np.uint8),
            "points3d": pred_pts3d,
            "depth": predictions["depth"][view_idx].squeeze(),
            "normal": normals,
            "mask": mask,
        }
    return processed


def update_view_selectors(processed_data: Dict[int, Dict]):
    if not processed_data:
        choices = ["View 1"]
    else:
        choices = [f"View {i + 1}" for i in processed_data.keys()]
    return (
        gr.Dropdown(choices=choices, value=choices[0]),
        gr.Dropdown(choices=choices, value=choices[0]),
        gr.Dropdown(choices=choices, value=choices[0]),
    )


def _get_view(processed_data: Dict[int, Dict], view_index: int):
    if not processed_data:
        return None
    keys = sorted(processed_data.keys())
    view_index = max(0, min(view_index, len(keys) - 1))
    return processed_data[keys[view_index]]


def update_depth_view(processed_data: Dict[int, Dict], view_index: int):
    data = _get_view(processed_data, view_index)
    if data is None:
        return None
    return colorize_depth(data["depth"], mask=data.get("mask"))


def update_normal_view(processed_data: Dict[int, Dict], view_index: int):
    data = _get_view(processed_data, view_index)
    if data is None:
        return None
    return colorize_normal(data["normal"], mask=data.get("mask"))


def update_measure_view(processed_data: Dict[int, Dict], view_index: int):
    data = _get_view(processed_data, view_index)
    if data is None:
        return None, []
    image = data["image"].copy()
    mask = data.get("mask")
    if mask is not None:
        invalid_mask = ~mask
        if invalid_mask.any():
            overlay_color = np.array([255, 220, 220], dtype=np.uint8)
            alpha = 0.5
            for c in range(3):
                image[:, :, c] = np.where(
                    invalid_mask,
                    (1 - alpha) * image[:, :, c] + alpha * overlay_color[c],
                    image[:, :, c],
                ).astype(np.uint8)
    return image, []


def populate_visualization_tabs(processed_data: Dict[int, Dict]):
    if not processed_data:
        return None, None, None, []
    depth_vis = update_depth_view(processed_data, 0)
    normal_vis = update_normal_view(processed_data, 0)
    measure_img, _ = update_measure_view(processed_data, 0)
    return depth_vis, normal_vis, measure_img, []


def navigate_view(processed_data, current_selector_value, direction, update_fn):
    if not processed_data:
        return "View 1", None
    try:
        current = int(current_selector_value.split()[1]) - 1
    except Exception:
        current = 0
    keys = sorted(processed_data.keys())
    new_idx = (current + direction) % len(keys)
    return f"View {new_idx + 1}", update_fn(processed_data, new_idx)


def navigate_depth_tab(processed_data, selector_value, direction):
    new_selector, vis = navigate_view(
        processed_data, selector_value, direction, update_depth_view
    )
    return new_selector, vis


def navigate_normal_tab(processed_data, selector_value, direction):
    new_selector, vis = navigate_view(
        processed_data, selector_value, direction, update_normal_view
    )
    return new_selector, vis


def navigate_measure_tab(processed_data, selector_value, direction):
    if not processed_data:
        return "View 1", None, []
    try:
        current = int(selector_value.split()[1]) - 1
    except Exception:
        current = 0
    keys = sorted(processed_data.keys())
    new_idx = (current + direction) % len(keys)
    new_selector = f"View {new_idx + 1}"
    img, _ = update_measure_view(processed_data, new_idx)
    return new_selector, img, []


def export_glb(
    predictions: Dict[str, np.ndarray],
    show_cam: bool,
    filter_black_bg: bool,
    filter_white_bg: bool,
    show_mesh: bool,
    conf_thres: float,
) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    glb_path = OUTPUT_DIR / (
        f"scene_cam{int(show_cam)}_mesh{int(show_mesh)}_"
        f"black{int(filter_black_bg)}_white{int(filter_white_bg)}_"
        f"conf{conf_thres:.1f}.glb"
    )
    glb = predictions_to_glb(
        predictions,
        filter_by_frames="All",
        show_cam=show_cam,
        mask_black_bg=filter_black_bg,
        mask_white_bg=filter_white_bg,
        as_mesh=show_mesh,
        conf_percentile=conf_thres if conf_thres > 0 else None,
    )
    glb.export(file_obj=str(glb_path))
    return str(glb_path)


def update_visualization(
    show_cam, filter_black_bg, filter_white_bg, show_mesh, conf_thres
):
    glb_path = export_glb(
        PREDICTIONS,
        show_cam=show_cam,
        filter_black_bg=filter_black_bg,
        filter_white_bg=filter_white_bg,
        show_mesh=show_mesh,
        conf_thres=conf_thres,
    )
    return glb_path, "可视化已更新。"


def update_all_views_on_filter_change(filter_black_bg, filter_white_bg):
    new_data = process_predictions_for_visualization(
        PREDICTIONS, PROCESSED_VIEWS, filter_black_bg, filter_white_bg
    )
    depth_vis = update_depth_view(new_data, 0)
    normal_vis = update_normal_view(new_data, 0)
    measure_img, _ = update_measure_view(new_data, 0)
    return new_data, depth_vis, normal_vis, measure_img, []


def measure(processed_data, measure_points, selector_value, event: gr.SelectData):
    if not processed_data:
        return None, [], "没有可用数据"
    try:
        view_idx = int(selector_value.split()[1]) - 1
    except Exception:
        view_idx = 0
    view = _get_view(processed_data, view_idx)
    if view is None:
        return None, [], "没有可用数据"

    point = (event.index[0], event.index[1])
    mask = view.get("mask")
    if (
        mask is not None
        and 0 <= point[1] < mask.shape[0]
        and 0 <= point[0] < mask.shape[1]
        and not mask[point[1], point[0]]
    ):
        image, _ = update_measure_view(processed_data, view_idx)
        return (
            image,
            measure_points,
            '<span style="color: red; font-weight: bold;">选中区域无深度信息。</span>',
        )

    measure_points.append(point)
    image, _ = update_measure_view(processed_data, view_idx)
    image = image.copy()
    points3d = view["points3d"]
    depth = view["depth"]

    for p in measure_points:
        if 0 <= p[0] < image.shape[1] and 0 <= p[1] < image.shape[0]:
            image = cv2.circle(image, p, radius=5, color=(255, 0, 0), thickness=2)

    depth_msg = ""
    for idx, p in enumerate(measure_points):
        if 0 <= p[1] < depth.shape[0] and 0 <= p[0] < depth.shape[1]:
            depth_msg += f"- **P{idx + 1} depth: {depth[p[1], p[0]]:.2f}m**\n"

    if len(measure_points) == 2:
        p1, p2 = measure_points
        if (
            0 <= p1[0] < image.shape[1]
            and 0 <= p1[1] < image.shape[0]
            and 0 <= p2[0] < image.shape[1]
            and 0 <= p2[1] < image.shape[0]
        ):
            image = cv2.line(image, p1, p2, color=(255, 0, 0), thickness=2)
        distance_text = "- **Distance: 无法计算**"
        if (
            0 <= p1[1] < points3d.shape[0]
            and 0 <= p1[0] < points3d.shape[1]
            and 0 <= p2[1] < points3d.shape[0]
            and 0 <= p2[0] < points3d.shape[1]
        ):
            distance = np.linalg.norm(points3d[p1[1], p1[0]] - points3d[p2[1], p2[0]])
            distance_text = f"- **Distance: {distance:.2f}m**"
        measure_points = []
        return image, measure_points, depth_msg + distance_text

    return image, measure_points, depth_msg


def initialize_demo():
    global PROCESSED_VIEWS, PREDICTIONS, PROCESSED_DATA, DEFAULT_GLB_PATH
    start_time = time.time()
    raw_views = load_fixed_views()
    processed_views = preprocess_views(raw_views)
    outputs = run_inference(processed_views)
    predictions = build_predictions(outputs)
    processed_data = process_predictions_for_visualization(predictions, processed_views)
    glb_path = export_glb(
        predictions,
        show_cam=True,
        filter_black_bg=False,
        filter_white_bg=False,
        show_mesh=True,
        conf_thres=0,
    )
    PROCESSED_VIEWS = processed_views
    PREDICTIONS = predictions
    PROCESSED_DATA = processed_data
    DEFAULT_GLB_PATH = glb_path
    elapsed = time.time() - start_time
    return len(processed_views), elapsed


num_views, init_time = initialize_demo()


theme = get_gradio_theme()
header_logo = get_logo_base64()

with gr.Blocks(theme=theme, css=GRADIO_CSS) as demo:
    processed_data_state = gr.State(value=PROCESSED_DATA)
    measure_points_state = gr.State(value=[])

    gr.HTML(get_header_html(header_logo))
    gr.HTML(get_description_html())

    gr.Markdown(f"已加载 {num_views} 张图像，初始推理耗时约 {init_time:.2f} 秒。")

    depth_vis, normal_vis, measure_img, _ = populate_visualization_tabs(PROCESSED_DATA)
    depth_selector, normal_selector, measure_selector = update_view_selectors(
        PROCESSED_DATA
    )

    with gr.Row():
        with gr.Column(scale=4):
            reconstruction_output = gr.Model3D(
                value=DEFAULT_GLB_PATH,
                height=520,
                zoom_speed=0.5,
                pan_speed=0.5,
                clear_color=[0.0, 0.0, 0.0, 0.0],
            )
            log_output = gr.Markdown("初始重建完成。")

            with gr.Tabs():
                with gr.Tab("Depth"):
                    with gr.Row(elem_classes=["navigation-row"]):
                        prev_depth_btn = gr.Button("◀ Previous", size="sm", scale=1)
                        depth_view_selector = gr.Dropdown(
                            choices=depth_selector.choices,
                            value=depth_selector.value,
                            label="Select View",
                            scale=2,
                            interactive=True,
                        )
                        next_depth_btn = gr.Button("Next ▶", size="sm", scale=1)
                    depth_map = gr.Image(
                        value=depth_vis,
                        type="numpy",
                        label="Colorized Depth Map",
                        format="png",
                        interactive=False,
                    )
                with gr.Tab("Normal"):
                    with gr.Row(elem_classes=["navigation-row"]):
                        prev_normal_btn = gr.Button("◀ Previous", size="sm", scale=1)
                        normal_view_selector = gr.Dropdown(
                            choices=normal_selector.choices,
                            value=normal_selector.value,
                            label="Select View",
                            scale=2,
                            interactive=True,
                        )
                        next_normal_btn = gr.Button("Next ▶", size="sm", scale=1)
                    normal_map = gr.Image(
                        value=normal_vis,
                        type="numpy",
                        label="Normal Map",
                        format="png",
                        interactive=False,
                    )
                with gr.Tab("Measure"):
                    gr.Markdown(MEASURE_INSTRUCTIONS_HTML)
                    with gr.Row(elem_classes=["navigation-row"]):
                        prev_measure_btn = gr.Button("◀ Previous", size="sm", scale=1)
                        measure_view_selector = gr.Dropdown(
                            choices=measure_selector.choices,
                            value=measure_selector.value,
                            label="Select View",
                            scale=2,
                            interactive=True,
                        )
                        next_measure_btn = gr.Button("Next ▶", size="sm", scale=1)
                    measure_image = gr.Image(
                        value=measure_img,
                        type="numpy",
                        show_label=False,
                        format="webp",
                        interactive=False,
                        sources=[],
                    )
                    gr.Markdown("**提示：** 灰色区域表示无效深度，无法进行测量。")
                    measure_text = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### 可视化选项")
            conf_thres = gr.Slider(
                minimum=0,
                maximum=100,
                value=0,
                step=0.1,
                label="Confidence Threshold Percentile",
                interactive=True,
            )
            show_cam = gr.Checkbox(label="Show Camera", value=True)
            show_mesh = gr.Checkbox(label="Show Mesh", value=True)
            filter_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
            filter_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # 事件绑定
    prev_depth_btn.click(
        fn=lambda data, selector: navigate_depth_tab(data, selector, -1),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_view_selector, depth_map],
    )
    next_depth_btn.click(
        fn=lambda data, selector: navigate_depth_tab(data, selector, 1),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_view_selector, depth_map],
    )
    depth_view_selector.change(
        fn=lambda data, selector: update_depth_view(data, int(selector.split()[1]) - 1),
        inputs=[processed_data_state, depth_view_selector],
        outputs=[depth_map],
    )

    prev_normal_btn.click(
        fn=lambda data, selector: navigate_normal_tab(data, selector, -1),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_view_selector, normal_map],
    )
    next_normal_btn.click(
        fn=lambda data, selector: navigate_normal_tab(data, selector, 1),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_view_selector, normal_map],
    )
    normal_view_selector.change(
        fn=lambda data, selector: update_normal_view(
            data, int(selector.split()[1]) - 1
        ),
        inputs=[processed_data_state, normal_view_selector],
        outputs=[normal_map],
    )

    prev_measure_btn.click(
        fn=lambda data, selector: navigate_measure_tab(data, selector, -1),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_view_selector, measure_image, measure_points_state],
    )
    next_measure_btn.click(
        fn=lambda data, selector: navigate_measure_tab(data, selector, 1),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_view_selector, measure_image, measure_points_state],
    )
    measure_view_selector.change(
        fn=lambda data, selector: update_measure_view(
            data, int(selector.split()[1]) - 1
        ),
        inputs=[processed_data_state, measure_view_selector],
        outputs=[measure_image, measure_points_state],
    )

    measure_image.select(
        fn=measure,
        inputs=[processed_data_state, measure_points_state, measure_view_selector],
        outputs=[measure_image, measure_points_state, measure_text],
    )

    filter_black_bg.change(
        fn=update_all_views_on_filter_change,
        inputs=[filter_black_bg, filter_white_bg],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
            measure_image,
            measure_points_state,
        ],
    ).then(
        fn=update_visualization,
        inputs=[show_cam, filter_black_bg, filter_white_bg, show_mesh, conf_thres],
        outputs=[reconstruction_output, log_output],
    )

    filter_white_bg.change(
        fn=update_all_views_on_filter_change,
        inputs=[filter_black_bg, filter_white_bg],
        outputs=[
            processed_data_state,
            depth_map,
            normal_map,
            measure_image,
            measure_points_state,
        ],
    ).then(
        fn=update_visualization,
        inputs=[show_cam, filter_black_bg, filter_white_bg, show_mesh, conf_thres],
        outputs=[reconstruction_output, log_output],
    )

    for control in [conf_thres, show_cam, show_mesh]:
        control.change(
            fn=update_visualization,
            inputs=[show_cam, filter_black_bg, filter_white_bg, show_mesh, conf_thres],
            outputs=[reconstruction_output, log_output],
        )

    gr.HTML(get_acknowledgements_html())

demo.queue(max_size=20).launch(show_error=True, share=False, ssr_mode=False)
