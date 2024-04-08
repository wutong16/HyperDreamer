#!/usr/bin/env bash

set -x

case=$1
exp_dir=$2
pyargs="${@:3}"

python main.py -O --image ${case} --workspace ${exp_dir} --iters 5000 ${pyargs}
