import argparse
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os 
from transformers import CLIPVisionModel, CLIPImageProcessor
from pathlib import Path
from vit_rollout_clip import CLIPVITAttentionRollout
from vit_grad_rollout_clip import CLIPViTAttentionRollout


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--output_root', type=str, default='./CLIP-ViT',
                        help='Top-level directory under which results will be saved.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name of subfolder under output_root. '
                        'If omitted, the parent folder name of image_path is used.')
    parser.add_argument('--wide_split', type=int, default=0,
                        help='If >0, treat image_path as a wide panorama and split '
                            'horizontally into this many equal (last takes remainder) crops.')
    parser.add_argument('--wide_fov_h', type=float, default=90.0,
                        help='Horizontal FOV (deg) for wide projection mode.')
    parser.add_argument('--wide_fov_v', type=float, default=120.0,
                        help='Vertical FOV (deg) for wide projection mode.')
    parser.add_argument('--wide_out_w', type=int, default=480,
                        help='Projected view width (pre-CLIP).')
    parser.add_argument('--wide_out_h', type=int, default=640,
                        help='Projected view height (pre-CLIP).')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    parser.add_argument('--save_id', type=int, default=0,
                        help="Image number for iteration")
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args
def _generate_projection_grid(fov_h_deg, fov_v_deg, yaw_deg, pitch_deg,
                              width, height, device):
    """
    Create direction rays for a perspective camera with given FOV & yaw/pitch,
    return lon/lat (spherical angles) per output pixel.
    """
    fov_h = np.radians(fov_h_deg)
    fov_v = np.radians(fov_v_deg)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    # Focal lengths in pixel units
    fx = 0.5 * width / np.tan(fov_h / 2)
    fy = 0.5 * height / np.tan(fov_v / 2)

    i, j = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing='xy'
    )
    x = (i - width / 2) / fx
    y = -(j - height / 2) / fy
    z = torch.ones_like(x)

    # Normalize dirs
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm

    dirs = torch.stack([x, y, z], dim=-1)

    # Rotations: yaw (Y), pitch (X)
    Ry = torch.tensor([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=torch.float32, device=device)

    Rx = torch.tensor([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ], dtype=torch.float32, device=device)

    R = Rx @ Ry
    dirs = dirs @ R.T

    lon = torch.atan2(dirs[..., 0], dirs[..., 2])  # [-pi, pi]
    lat = torch.asin(torch.clamp(dirs[..., 1], -1.0, 1.0))  # [-pi/2, pi/2]
    return lon, lat


def _spherical_to_equirectangular_coords(lon, lat, width, height):
    """
    Map lon/lat to equirectangular pixel coords (normalized grid for grid_sample).
    """
    u = (lon + np.pi) / (2 * np.pi) * width      # [0, W)
    v = (np.pi/2 - lat) / np.pi * height         # [0, H)

    u_norm = (u / width) * 2 - 1                 # [-1,1]
    v_norm = (v / height) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1)
    return grid

def _fix_orientation_cw_mirror(pil_img):
    """
    현재 추출된 이미지가 '왼쪽으로 90도 회전 + 좌우대칭' 되어 보이는 문제를
    반대로 보정: 시계방향(CW) 90도 회전 후 좌우반전.
    """
    return pil_img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)

def _extract_4_views_from_equirect(img_path, device, hfov=90, vfov=120,
                                   w_out=480, h_out=640):
    """
    Load equirect panorama, return list of (PIL_Image, base_name) for 4 cardinal views.
    Order: front(0), right(1), back(2), left(3) -> wide_angle_i naming.
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)  # H,W,3
    H_in, W_in = img_np.shape[:2]

    # to tensor [1,3,H,W] float
    img_tensor = torch.from_numpy(img_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    views = {"front": 0, "right": 90, "back": 180, "left": 270}
    outputs = {}

    for name, yaw in views.items():
        lon, lat = _generate_projection_grid(hfov, vfov, yaw, 0, w_out, h_out, device)
        grid = _spherical_to_equirectangular_coords(lon, lat, W_in, H_in).unsqueeze(0)
        # grid_sample expects grid shape [N,H_out,W_out,2] with last dim (x,y)
        # our stack above produced shape [1,w_out,h_out,2]? Actually we made indexing 'xy':
        # We used meshgrid width x height; that yields grid(W,H,2) but torch grid_sample wants (H,W).
        # We need to permute.
        grid = grid.permute(0, 2, 1, 3)  # (1,h_out,w_out,2)
        out = F.grid_sample(
            img_tensor, grid, mode='bilinear',
            align_corners=True, padding_mode='border'
        )
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        outputs[name] = out_np

    front = _fix_orientation_cw_mirror(Image.fromarray(outputs["front"]))
    right = _fix_orientation_cw_mirror(Image.fromarray(outputs["right"]))
    back  = _fix_orientation_cw_mirror(Image.fromarray(outputs["back"]))
    left  = _fix_orientation_cw_mirror(Image.fromarray(outputs["left"]))

    return [
        (front, "wide_angle_0"),
        (right, "wide_angle_1"),
        (back,  "wide_angle_2"),
        (left,  "wide_angle_3"),
    ]

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def disable_fused_attn(model):
    for block in model.blocks:
        block.attn.fused_attn = False

if __name__ == '__main__':
    args = get_args()
    img_path = Path(args.image_path).expanduser().resolve()
    folder_name = args.run_name if args.run_name is not None else img_path.parent.name
    view_name = img_path.stem
    out_dir = Path(args.output_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    target_image = []
    if args.wide_split > 0 :
        wide = Image.open(img_path).convert("RGB")
        W, H = wide.size
        step = W // args.wide_split  # 마지막 조각은 remainder 포함
        for i in range(args.wide_split):
            left = i * step
            right = (i + 1) * step if i < args.wide_split - 1 else W
            crop = wide.crop((left, 0, right, H))
            crop = crop.resize((336, 336))  # CLIP ViT-L/14-336
            target_image.append((crop, f"wide_angle_{i}"))
    # if args.wide_split > 0:
    #     device = 'cuda' if args.use_cuda else 'cpu'
    #     target_image = _extract_4_views_from_equirect(
    #         img_path,
    #         device=device,
    #         hfov=args.wide_fov_h,
    #         vfov=args.wide_fov_v,
    #         w_out=args.wide_out_w,
    #         h_out=args.wide_out_h,
    #     )
    else: 
        img = Image.open(img_path).convert("RGB").resize((336, 336))
        target_image.append((img, view_name))
        
    for img, base_name in target_image:
        model = CLIPVisionModel.from_pretrained(       # Vision-only Video-LLaMA2 uses clip-vit as visual encoder. See https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV#vision-only-checkpoints
            "openai/clip-vit-large-patch14-336",     # Solely loading Vision Model of clip-vit
            torch_dtype="auto",
            low_cpu_mem_usage=True)
        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        model.eval()

        # disable_fused_attn(model)
        
        if args.use_cuda:
            model = model.cuda()

        inputs = clip_processor(images=img, return_tensors="pt")
        input_tensor = inputs["pixel_values"]    
        
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = CLIPVITAttentionRollout(model, head_fusion=args.head_fusion, 
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = out_dir / "{}_attention_rollout_{:.3f}_{}_{}.png".format(base_name, args.discard_ratio, args.head_fusion, args.save_id)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = CLIPViTAttentionRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name =  out_dir / "{}_grad_rollout_{}_{:.3f}_{}_{}.png".format(base_name, args.category_index,
                args.discard_ratio, args.head_fusion, args.save_id)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        in_name = out_dir / f"{base_name}_input.png"
        cv2.imwrite(str(in_name), np_img)
        cv2.imwrite(str(name), mask)
        print(f"[✓] Saved: input.png  {name}")