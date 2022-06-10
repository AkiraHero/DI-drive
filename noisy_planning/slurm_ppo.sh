#!/bin/bash
set -x


JOB_NAME=hello_world
GPUS_PER_NODE=1
GPUS=${GPUS_PER_NODE}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=12 \
    --kill-on-bad-exit=1 \
    --partition=shlab_ap_ll \
    sh st_ppo.sh $1
