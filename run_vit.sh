for IMG_ID in {1..8}
do
    python3 vit_explain_clip.py \
        --image_path ./data/img_$IMG_ID.jpg \
        --head_fusion max \
        --discard_ratio 0.5 \
        --save_id $IMG_ID \
        # --category_index 559 \

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