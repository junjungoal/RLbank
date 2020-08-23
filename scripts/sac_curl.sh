#!/bin/bash -x
gpu=$1
seed=$2
algo='sac'
prefix="SAC.CURL"
# env="Reacher-v2"
# env="Hopper-v2"
env="walker"
task_name='walk'
debug="False"
log_root_dir="./logs"
entity="junyamada107"
project='rlgarage'
wandb='True'
batch_size='128'
buffer_size='100000'
policy='cnn'
frame_stack='3'
log_interval='1'
evaluate_interval='10'
rl_hid_size='1024'
init_step='1000'

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
    --frame_stack $frame_stack \
    --log_interval $log_interval \
    --evaluate_interval $evaluate_interval \
    --rl_hid_size $rl_hid_size \
    --init_step $init_step \
    --task_name $task_name
