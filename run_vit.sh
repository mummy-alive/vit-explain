#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=./data
OUT_ROOT=./CLIP-ViT 
SPLITS=4

for d in "${DATA_ROOT}"/*/; do
    folder=$(basename "$d")              # 예: seq001, car_A, ...
    echo "[*] Processing folder: $folder"

    for view in view_back view_front view_left view_right; do
        img="$d/${view}.jpg"
        if [[ ! -f "$img" ]]; then
            echo "    [!] Missing $img (skip)"
            continue
        fi

        python3 vit_explain_clip.py \
            --image_path "$img" \
            --output_root "$OUT_ROOT" \
            --run_name "$folder" \
            --head_fusion max \
            --discard_ratio 0.7
            # --category_index 559   # 필요 시 주석 해제
    done
    
    # Wide image 처리
    wide="$d/${folder}.jpg"
    if [[ -f "$wide" ]]; then
        echo "    [+] Wide image found: $wide (split into ${SPLITS})"
        python3 vit_explain_clip.py \
            --image_path "$wide" \
            --output_root "$OUT_ROOT" \
            --run_name "$folder" \
            --head_fusion max \
            --discard_ratio 0.7 \
            --wide_split "$SPLITS"
            # --category_index 559
    fi
done
    # model이 deit - vit_explain.py
    # model이 video-llama2 - vit_explain.py
    
    # head_fusion: <mean, min or max>
    # discard_ratio: 0~1 사이 숫자
    # category_index: 카테고리 인덱스
    #   532=dining table, board
    #   559=t
    #   831=studio couch, day bed
    #   883=vase
    #   