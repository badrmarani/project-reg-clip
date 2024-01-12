#!/bin/bash

num_repeats=1

for i in $(seq 1 $num_repeats)
do
    echo Run $i / $num_repeats
    python main.py \
        --device 2 \
        --max-epochs 20 \
        --save-dir "logs/" \
        --model-name "resnet18" \
        --num-classes 1 \
        --pretrained \
        --dataset "celeba" \
        --root-dir "../datasets/" \
        --spurious-label "Male" \
        --stratified-sampling \
        --use-image-captions \
        --batch-size 128 \
        --num-workers 4 \
        --optimizer-name "AdamW" \
        --lr 1e-5 \
        --weight-decay 0.0 \
        --momentum 0.9 \
        --normalize-clip-loss
done
