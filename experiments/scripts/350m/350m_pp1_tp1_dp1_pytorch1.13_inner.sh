#!/bin/bash

#mkdir -p /local/fcc/pytorch
#cp /home/u11890/work/MyPytorch.tar /local/fcc/
#cd /local/fcc
#tar xf MyPytorch.tar
#source /local/fcc/inst/venv/bin/activate
cd /home/u11890/work/flush-denormal/DeepSpeedFugaku

# Change for multinode config
CPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
export TIMER="timer.350m_pp1_tp1_dp1_pytorch1.13"
CHECKPOINT_PATH=checkpoints/350m_pp1_mp1_dp1_pytorch1.13/
INPUT_PREFIX=dataset
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=dataset/codeparrot_content_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

output_path="jobs/350m_pp1_mp1_dp1_pytorch1.13/outs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc"
DATA_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48
export LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
export OMP_WAIT_POLICY=ACTIVE

python pretrain_gpt.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 2 \
    --lr-decay-iters 320000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --vocab-file $INPUT_PREFIX/$VOCAB_FILE \
    --merge-file $INPUT_PREFIX/$MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1 \
    --distributed-backend mpi \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --lr-warmup-fraction .01 \
    --checkpoint-activations \
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --no-cuda \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS
