#!/bin/bash
#
# $0 <PP> <TP> <DP> <GLOBAL_BATCH> <MYGEMM> <MICRO_BATCH_SIZE> <NUM_ATTENTION_HEADS> <GEMM_DEBUG_FLAG>

RANK=$PMIX_RANK
SIZE=$PJM_NODE

# $PP $TP $DP $NITER $GB $MYGEMM $MB $AHEADS $GEMM_DEBUG
PIPELINE_MODEL_PARALLEL_SIZE=$1 && shift
TENSOR_MODEL_PARALLEL_SIZE=$1 && shift
DATA_PARALLEL_SIZE=$1 && shift
GLOBAL_BATCH=$1 && shift
export MYGEMM=$1 && shift
MICRO_BATCH_SIZE=$1 && shift
NUM_ATTENTION_HEADS=$1 && shift
if [ $1 -eq 0 ]; then
  unset A64FX_CBLAS_SGEMM_BATCH_DEGBUG
else
  export A64FX_CBLAS_SGEMM_BATCH_DEGBUG=$1
fi && shift

# PyTorch 1.13
PYTORCH_ENV_PATH=/vol0005/mdt3/share/hp230254/pytorch
PYTORCH_ENV=1695121168.286518215.fcc.pytorch.y.r1.13_for_a64fx.tar.gz

mkdir -p /local/fcc/pytorch
cp ${PYTORCH_ENV_PATH}/${PYTORCH_ENV} /worktmp
cd /local/fcc
tar xfz /worktmp/${PYTORCH_ENV}
rm /worktmp/${PYTORCH_ENV}
source /local/fcc/inst/venv/bin/activate

# PyTorch 1.10.1
#source /data/hp190122/share/PyTorch-1.10.1/env.src
#export PYTHONUSERBASE=$HOME/work/.local
#export PATH=$PATH:$PYTHONUSERBASE/bin

cd ${HOME}/work/DeepSpeedFugaku

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

# Change for multinode config
CPUS_PER_NODE=1
NNODES=$(( $PIPELINE_MODEL_PARALLEL_SIZE * $TENSOR_MODEL_PARALLEL_SIZE * $DATA_PARALLEL_SIZE))
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
export TIMER="timer.1.3b_pp_tp_dp_pytorch1.13_8215tgz"
CHECKPOINT_PATH=checkpoints/1.3b_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k/ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}
VOCAB_FILE=$HOME/work/DeepSpeedFugaku/tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model
#INPUT_PREFIX=dataset
#VOCAB_FILE=gpt2-vocab.json
#MERGE_FILE=gpt2-merges.txt

mkdir -p $CHECKPOINT_PATH

# dataset setting
##
#DATA_PATH=data/codeparrot/codeparrot_content_document
##
DATASET_PATH=/data/hp190122/share/dataset/wikipedia/binarized/v2_1-code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k

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

TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

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

export LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
#export OMP_WAIT_POLICY=ACTIVE
#if [ $RANK -eq 0 ]; then
#  echo "# DEBUG flag=" $A64FX_CBLAS_SGEMM_BATCH_DEGBUG
#  echo "# GLOBAL_BATCH=" $GLOBAL_BATCH
#  echo "# MYGEMM=" $MYGEMM
#fi

# ref. https://www.fugaku.r-ccs.riken.jp/faq/20210428_01 , https://www.fugaku.r-ccs.riken.jp/bug/20210428_02
export UTOFU_SWAP_PROTECT=1

numactl -m 4-7 -N 4-7 python pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH} \
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
    --cpu-optimizer \
    --cpu-torch-adam \
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
  --wandb-name "1.3B-pp${PIPELINE_MODEL_PARALLEL_SIZE}-tp${TENSOR_MODEL_PARALLEL_SIZE}-dp${DATA_PARALLEL_SIZE}-gb${GLOBAL_BATCH}-GEMM${MYGEMM}-ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}" \
  --use-timer \
  --use-flush-denormal

  #--log-batch-size-to-tensorboard \
  #--log-validation-ppl-to-tensorboard \
  #--log-timers-to-tensorboard \
  #--log-optimizer-states-to-tensorboard \
  #  $TENSORBOARD_ARGS \
    #--merge-file $INPUT_PREFIX/$MERGE_FILE \
    #--train-iters ${TRAIN_ITERS} \
    #--lr-decay-iters 320000 \
    #--wandb-name "1.3B-pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_gb${GLOBAL_BATCH}_GEMM${MYGEMM}"

echo `date` ALL_DONE
