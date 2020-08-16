#!/bin/bash -x
gpu=$1
seed=$2
algo='ppo'
prefix="ppo"
env="Reacher-v2"
# env="Hopper-v2"
# env="Walker2d-v2"
debug="False"
log_root_dir="./logs"
entity="junyamada107"
project='rlgarage'
wandb='True'
multiprocessing="True"
rollout_length="128"
num_processes='16'
--algo ppo --multiprocessing True --rollout_length 32

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
    --multiprocessing $multiprocessing \
    --num_processes $num_processes \
    --rollout_length $rollout_length
