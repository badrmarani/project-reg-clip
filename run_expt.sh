#!/bin/bash

num_repeats=1

for i in $(seq 1 $num_repeats)
do
    echo Run $i / $num_repeats
    python main.py\
        --max-epochs 1\
        --save-dir "logs/"\
        --model-name "resnet18"\
        --num-classes 1\
        --pretrained\
        --dataset "celeba"\
        --root-dir "/home/jupyterlab/datasets"\
        --spurious-label "Male"\
        --stratified-sampling\
        --batch-size 128\
        --num-workers 2\
        --optimizer-name "SGD"\
        --lr 1e-5\
        --weight-decay 1.0\
        --momentum 0.9
done
