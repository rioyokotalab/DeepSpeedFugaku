#!/bin/bash

param_name='30b_pp12_tp4_dp256_pytorch1.13'
NNODES=12288
DATA_PARALLEL_SIZE=256
PIPELINE_MODEL_PARALLEL_SIZE=12
TENSOR_MODEL_PARALLEL_SIZE=4

mkdir -p /local/fcc/pytorch
cd /local/fcc
tar xf /home/u11890/work/1693389241.318550480.fcc.pytorch.y.r1.13_for_a64fx.tar
source /local/fcc/inst/venv/bin/activate
cd /home/u11890/work/timer/DeepSpeedFugaku

# Change for multinode config
CPUS_PER_NODE=1
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
export TIMER="timer.${param_name}"
CHECKPOINT_PATH=checkpoints/${param_name}/
INPUT_PREFIX=dataset
VOCAB_FILE=tokenizer/models/ver2/code20k_en40k_ja60k.ver2.1.model
DATA_PATH=dataset/codeparrot_content_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48
export LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
#export OMP_WAIT_POLICY=ACTIVE

numactl -m 4-7 -N 4-7 python pretrain_gpt.py \
    --num-layers 48 \
    --hidden-size 6912 \
    --num-attention-heads 72 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 5 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tokenizer-type JapaneseSentencePiece \
    --vocab-file $VOCAB_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend mpi \
    --init-method-std 0.013 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --cpu-optimizer \
    --cpu-torch-adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 5 \
    --no-cuda \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    --use-timer \
    --use-flush-denormal
