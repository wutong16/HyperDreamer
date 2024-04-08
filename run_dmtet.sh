#!/usr/bin/env bash

set -x

case=$1
init_exp_dir=$2
exp_dir=$3
pyargs="${@:4}"

python main.py --cuda_ray --image ${case} --workspace ${exp_dir} \
    --dmtet --init_with ${init_exp_dir}/checkpoints/df.pth \
    --use_svbrdf --global_sam --derender_reg --lambda_super_reso 200 ${pyargs}
