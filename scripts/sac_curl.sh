#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="SAC.CURL"
# env="Reacher-v2"
# env="Hopper-v2"
env="cheetah"
debug="False"
log_root_dir="./logs"
entity="junyamada107"
project='rlgarage'
wandb='True'
batch_size='32'
buffer_size='100000'
policy='cnn'
frame_stack='3'


python -m main \
    --log_root_dir $log_root_dir \
    --wandb True \
    --prefix $prefix \
    --env $env \
    --gpu $gpu \
    --debug $debug \
    --algo $algo \
    --seed $seed \
    --wandb $wandb \
    --entity $entity \
    --project $project \
    --buffer_size $buffer_size \
    --batch_size $batch_size \
    --policy $policy \
    --frame_stack $frame_stack
