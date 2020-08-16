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
rollout_length="2048"
num_processes='1'
activation="tanh"
log_interval="1"
evaluate_interval="10"
ppo_epoch='10'
batch_size="64"
max_global_step="1000000"

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
    --rollout_length $rollout_length \
    --activation $activation \
    --log_interval $log_interval \
    --evaluate_interval $evaluate_interval \
    --ppo_epoch $ppo_epoch \
    --batch_size $batch_size \
    --max_global_step $max_global_step
