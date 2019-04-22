#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET_NAME=ResidualGRUNet
EXP_DETAIL=default_model
OUT_PATH='./output/'$NET_NAME/$EXP_DETAIL
LOG="$OUT_PATH/log.`date +'%Y-%m-%d_%H-%M-%S'`"

# Make the dir if it not there
mkdir -p $OUT_PATH
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

export THEANO_FLAGS="floatX=float32,device=cuda0,optimizer_including=cudnn,assert_no_cpu_op='raise'"

python3 main.py \
      --batch-size 20 \
      --iter 20000 \
      --out $OUT_PATH \
      --model $NET_NAME \
      ${*:1}

python3 main.py \
      --test \
      --batch-size 1 \
      --out $OUT_PATH \
      --weights $OUT_PATH/checkpoint.tar \
      --model $NET_NAME \
      ${*:1}
