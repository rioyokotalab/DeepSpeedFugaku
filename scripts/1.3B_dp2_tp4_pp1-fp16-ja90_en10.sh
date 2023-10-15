#!/bin/bash
#YBATCH -r dgx-a100_8
#SBATCH --job-name=gpt
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --output outputs/%j.out
#SBATCH --error errors/%j.err
. /etc/profile.d/modules.sh
module load cuda/11.7
module load cudnn/cuda-11.x/8.9.0
module load nccl/cuda-11.7/2.14.3
module load openmpi/4.0.5

set -e

source .env/bin/activate

# Tokenizer setting
CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=60

# Change for multinode config
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

# model setting
lr=2.0e-4
min_lr=1.0e-6
init_std=0.013
sequence_length=2048

# distributed settings
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
MASTER_NODE=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

# dataset setting
JA_PERTCENT=90
EN_PERTCENT=10
CODE_PERTCENT=0

# dataset weight setting
ja_wiki_weight=0.01014874486
en_wiki_weight=0.03344558481
ja_cc_weight=0.02341713829
en_pile_weight=0.001478987004
code_stack_weight=0

# training setting
train_token_in_billion=159
train_tokens=$(echo "$train_token_in_billion * 1000 * 1000 * 1000" | bc)
train_tokens=$(echo "$train_tokens/1" | bc)

lr_warmup_tokens_in_billion=1.59
lr_warmup_tokens=$(echo "$lr_warmup_tokens_in_billion * 1000 * 1000 * 1000" | bc)
lr_warmup_tokens=$(echo "$lr_warmup_tokens/1" | bc)

# same as megatron deepspeed setting
lr_decay_tokens_in_billion=${train_token_in_billion}
lr_decay_tokens=${train_tokens}

train_samples=$((300 * 1000000000 * 2 / ${sequence_length}))
exit_duration=30000000

# dataset path
DATASET_PATH="/mnt/nfs/Users/tn/fugaku/datasets/wikipedia/binarized/v2_1-code20k_en40k_ja60k"

DATA_PATH=""

DATA_PATH="${DATA_PATH} ${ja_wiki_weight} ${DATASET_PATH}/ja_wiki_text_document" # wiki (ja)

for i in {0..44}; do
  # pile (en)
  DATA_PATH="${DATA_PATH} ${en_pile_weight} ${DATASET_PATH}/en_pile${i}_text_document"
done

DATA_PATH="${DATA_PATH} ${en_wiki_weight} ${DATASET_PATH}/en_wiki_text_document" # wiki (en)

for i in {0..37}; do
  DATA_PATH="${DATA_PATH} ${ja_cc_weight} ${DATASET_PATH}/ja_cc${i}_text_document" # cc (ja)
done

vocab_path="llm-ja-tokenizer/models/ver2/code20k_en40k_ja60k.ver2.1.model"

# distributed setting
DATA_PARALLEL_SIZE=8

PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=4
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

# checkpoint
CHECKPOINT_PATH="/mnt/nfs/Users/kazuki/megatron-deepspeed/checkpoint/fugaku/1.3B-llm-jp-dataset-fp16-tp4-ja90_en10"

mkdir -p $CHECKPOINT_PATH

mpirun -np $WORLD_SIZE -npernode $GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_NODE \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
  python pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --micro-batch-size 2 \
  --global-batch-size 512 \
  --seq-length $sequence_length \
  --max-position-embeddings $sequence_length \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std ${init_std} \
  --lr-decay-tokens ${lr_decay_tokens} \
  --lr-warmup-tokens ${lr_warmup_tokens} \
  --train-tokens ${train_tokens} \
  --train-samples ${train_samples} \
  --data-path $DATA_PATH \
  --tokenizer-type JapaneseSentencePiece \
  --vocab-file $vocab_path \
  --data-impl mmap \
  --seed 1234 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr $lr \
  --min-lr $min_lr \
  --lr-decay-style cosine \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 50 \
  --eval-interval 100 \
  --eval-iters 10 \
  --checkpoint-activations \
  --use-cpu-initialization \
  --num-workers 0 \
  --fp16 \
  --use-mpi \
  $PARALLEL_ARGS \
  $TENSORBOARD_ARGS \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "1.3B-GPU-dp2-tp4-ja90_en10-llm-jp-dataset-fp16"
