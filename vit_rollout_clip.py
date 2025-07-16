import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion):
    fused_list = []
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            fused_list.append(attention_heads_fused)
    result = torch.eye(attentions[0].size(-1))

    for attention_heads_fused in fused_list:
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0

        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0*I)/2
        a = a / a.sum(dim=-1)

        result = torch.matmul(a, result)
        
    mask = result[0, 0 , 1 :]
    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    return mask    


class CLIPVITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def __call__(self, pixel_values):
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values,
                             output_attentions=True,
                             return_dict=True)
        self.attentions = [a.detach() for a in outputs.attentions]
        return rollout(self.attentions, self.discard_ratio, self.head_fusion)