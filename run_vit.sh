for IMG_ID in {1..5}
do
    python3 vit_explain.py \
        --image_path ./data/img_$IMG_ID.jpg \
        --head_fusion max \
        --discard_ratio 0 \
        --category_index 243 \
        --save_id $IMG_ID
done

    # head_fusion: <mean, min or max>
    # discard_ratio: 0~1 사이 숫자
    # category_index: 카테고리 인덱스