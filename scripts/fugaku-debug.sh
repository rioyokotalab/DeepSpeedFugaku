#!/bin/bash
#YBATCH -r a100_4
#SBATCH --job-name=gpt
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

source .env/bin/activate

JA_VOCAB_SIZE=10
EN_VOCAB_SIZE=40

# Change for multinode config
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0

export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_NODE=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

INPUT_PREFIX=dataset
VOCAB_FILE=tokenizer/models/cc100ja1GB_cc100en1GB/cc100_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K.symbolRemoved.vocab.reestimated.model
DATA_PATH=/home/kazuki/Documents/DeepSpeedFugaku/data/wikipedia/binarized/v1_ja${JA_VOCAB_SIZE}K_en${EN_VOCAB_SIZE}K/ja_wiki_text_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

DATA_PARALLEL_SIZE=4

PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"


mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_NODE \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --data-path $DATA_PATH \
  --tokenizer-type JapaneseSentencePiece \
  --vocab-file $VOCAB_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations \
  --use-cpu-initialization \
  --num-workers 0 \
  --use-mpi \
  $PARALLEL_ARGS \
  $TENSORBOARD_ARGS \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --log-timers-to-tensorboard \
  --log-optimizer-states-to-tensorboard \
  --wandb-name "GPU-old-version-no-zero-rng"
