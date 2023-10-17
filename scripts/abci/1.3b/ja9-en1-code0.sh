#!/bin/bash
#$ -l rt_F=256
#$ -l h_rt=48:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load hpcx/2.12

set -e

cd /groups/gaf51217/fujii/pretrain/DeepSpeedFugaku
source .env/bin/activate

# Tokenizer setting
CODE_VOCAB_SIZE=20
EN_VOCAB_SIZE=40
JA_VOCAB_SIZE=60

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
train_token=$(echo "$train_token_in_billion * 1000 * 1000 * 1000" | bc)
train_token=$(echo "$train_token/1" | bc)

# default megatron-deepspeed confgiraution is 3000 million, but they train model using 300 billion tokens. we use 159 billion tokens, so we set 1.59 billion tokens to lr-warmup-tokens.
lr_warmup_tokens_in_billion=1.59
lr_warmup_tokens=$(echo "$lr_warmup_tokens_in_billion * 1000 * 1000 * 1000" | bc)
lr_warmup_tokens=$(echo "$lr_warmup_tokens/1" | bc)

# same as megatron deepspeed setting
lr_decay_tokens_in_billion=${train_token_in_billion}
lr_decay_tokens=${train_token}

# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile

if [[ "$SGE_RESOURCE_TYPE" == "rt_F" ]]; then
  export NUM_GPU_PER_NODE=4
  NODE_TYPE="v100"
elif [[ "$SGE_RESOURCE_TYPE" == "rt_AF" ]]; then
  export NUM_GPU_PER_NODE=8
  NODE_TYPE="a100"
else
  echo "Unrecognized SGE_RESOURCE_TYPE: $SGE_RESOURCE_TYPE"
fi

NUM_NODES=$NHOSTS
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

mkdir -p ./hostfile

HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line
do
  echo "${line} slots=${NUM_GPU_PER_NODE}"
done < "$SGE_JOB_HOSTLIST" > "$HOSTFILE_NAME"

# checkpoint and tokenizer setting
CHECKPOINT_PATH=/groups/gaf51217/fujii/checkpoints/megatron-deepspeed/fugaku/1.3b_tp4_dp256_v2.1_code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k/ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}
VOCAB_FILE=tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model

mkdir -p $CHECKPOINT_PATH

# dataset setting
DATASET_PATH=/groups/gaf51217/fujii/datasets/megatron-deepspeed/v2_1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k

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

# stack (code)

# distributed setting
PIPELINE_MODEL_PARALLEL_SIZE=1
TENSOR_MODEL_PARALLEL_SIZE=4
DATA_PARALLEL_SIZE=$((${NUM_GPUS} / (${PIPELINE_MODEL_PARALLEL_SIZE} * ${TENSOR_MODEL_PARALLEL_SIZE})))

PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

export OMP_NUM_THREADS=48

# train samples
seq_len=2048
# we use another termination condition, train_tokens, instead of train_samples.
# but not using train_samples causes error. so we set train_samples to a large number.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

mpirun -np $NUM_GPUS \
  --npernode $NUM_GPU_PER_NODE \
  -hostfile $HOSTFILE_NAME \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none -map-by slot \
  -x PATH \
  python pretrain_gpt.py \
  --num-layers 24 \
  --hidden-size 2048 \
  --num-attention-heads 16 \
  --micro-batch-size 2 \
  --global-batch-size 512 \
  --seq-length $seq_len \
  --max-position-embeddings $seq_len \
  --train-tokens $train_token \
  --train-samples $train_samples \
  --lr-decay-tokens $lr_decay_tokens \
  --lr-warmup-tokens $lr_warmup_tokens \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --tokenizer-type JapaneseSentencePiece \
  --vocab-file $VOCAB_FILE \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend mpi \
  --init-method-std 0.013 \
  --lr 2.0e-4 \
  --min-lr 1.0e-6 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --log-interval 1 \
  --save-interval 1000 \
  --eval-interval 100 \
  --eval-iters 10 \
  --no-cuda \
  --checkpoint-activations \
  --use-cpu-initialization \
  --num-workers 0 \
  --no-load-rng \
  $PARALLEL_ARGS \
  $TENSORBOARD_ARGS \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --log-optimizer-states-to-tensorboard \
  --use-mpi \
  --wandb-name "abci-1.3b_gb512-ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}"
