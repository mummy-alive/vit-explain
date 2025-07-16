import argparse
import sys
import torch
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

def extract_vit(base_model): 
    for name, module in base_model.named_modules():
        print(name, type(module))

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
    out_dir = Path(f'CLIP-ViT/{str(args.category_index)}')        # 예: Path("243")
    out_dir.mkdir(parents=True, exist_ok=True)  
    model = CLIPVisionModel.from_pretrained(       # Vision-only Video-LLaMA2 uses clip-vit as visual encoder. See https://huggingface.co/DAMO-NLP-SG/VideoLLaMA2.1-7B-AV#vision-only-checkpoints
        "openai/clip-vit-large-patch14-336",     # Solely loading Vision Model of clip-vit
        torch_dtype="auto",
        low_cpu_mem_usage=True)
    clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    model.eval()

    # disable_fused_attn(model)
    
    if args.use_cuda:
        model = model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = Image.open(args.image_path)
    img = img.resize((224, 224))
    inputs = clip_processor(images=img, return_tensors="pt")
    input_tensor = inputs["pixel_values"]    
    
    if args.use_cuda:
        input_tensor = input_tensor.cuda()

    if args.category_index is None:
        print("Doing Attention Rollout")
        attention_rollout = CLIPVITAttentionRollout(model, head_fusion=args.head_fusion, 
            discard_ratio=args.discard_ratio)
        mask = attention_rollout(input_tensor)
        name = out_dir / "attention_rollout_{:.3f}_{}_{}.png".format(args.discard_ratio, args.head_fusion, args.save_id)
    else:
        print("Doing Gradient Attention Rollout")
        grad_rollout = CLIPViTAttentionRollout(model, discard_ratio=args.discard_ratio)
        mask = grad_rollout(input_tensor, args.category_index)
        name =  out_dir / "grad_rollout_{}_{:.3f}_{}_{}.png".format(args.category_index,
            args.discard_ratio, args.head_fusion, args.save_id)


    np_img = np.array(img)[:, :, ::-1]
    mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
    mask = show_mask_on_image(np_img, mask)
    cv2.imwrite("input.png", np_img)   # 원본 BGR 그대로
    cv2.imwrite(name, mask)
    print(f"[✓] Saved: input.png  {name}")