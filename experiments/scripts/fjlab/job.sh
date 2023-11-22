#!/bin/bash
#
# $0 <PYTORCH_TGZ> <MODEL> <TRAINDATA> <PP> <TP> <DP> <MB> <GLOBAL_BATCH> <MYGEMM> <GEMM_DEBUG_FLAG> <WANDBHEAD>

RANK=$PMIX_RANK
SIZE=$PJM_NODE

# $PYTORCH_TGZ $MODEL $TRAINDATA $PP $TP $DP $GB $MB $MYGEMM $GEMM_DEBUG
DEEPSPEEDFUGAKU_PATH=$1 && shift
PYTORCH_TGZ=$1 && shift
MODELSIZE=$1 && shift
TRAINDATA=$1 && shift
PIPELINE_MODEL_PARALLEL_SIZE=$1 && shift
TENSOR_MODEL_PARALLEL_SIZE=$1 && shift
DATA_PARALLEL_SIZE=$1 && shift
GLOBAL_BATCH=$1 && shift
MICRO_BATCH_SIZE=$1 && shift
LOGDIR=$1 && shift
export MYGEMM=$1 && shift
if [ $1 -eq 0 ]; then
  unset A64FX_CBLAS_SGEMM_BATCH_DEGBUG
else
  export A64FX_CBLAS_SGEMM_BATCH_DEGBUG=$1
fi && shift
WANDBHEAD=$1 && shift
SET_TRAIN_ITER=$1 && shift

## set model parameter
case ${MODELSIZE} in
  "125m")
    NUM_LAYERS=12
    HIDDEN_SIZE=768
    NUM_ATTENTION_HEADS=12
    ;;
  "1.3b")
    NUM_LAYERS=24
    HIDDEN_SIZE=2064
    NUM_ATTENTION_HEADS=16
    ;;
  "13b")
    NUM_LAYERS=40
    HIDDEN_SIZE=5160
    NUM_ATTENTION_HEADS=40
    ;;
  "30b")
    NUM_LAYERS=48
    HIDDEN_SIZE=6912
    NUM_ATTENTION_HEADS=96
    ;;
  *)
    echo "${MODELSIZE} is not supported"
    exit;;
esac

echo "Model: ${MODELSIZE}, hidden size: ${HIDDEN_SIZE}, # of attention heads: ${NUM_ATTENTION_HEADS}"

mkdir -p /local/fcc/pytorch
cd /local/fcc
#cp ${PYTORCH_TGZ} /worktmp
#tar xfz /worktmp/${PYTORCH_ENV}
#rm /worktmp/${PYTORCH_ENV}
tar xfz ${PYTORCH_TGZ}
source /local/fcc/inst/venv/bin/activate

cd ${DEEPSPEEDFUGAKU_PATH}

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
#train_token_in_billion=159
train_token_in_billion=206
train_token=$(echo "$train_token_in_billion * 1000 * 1000 * 1000" | bc)
train_token=$(echo "$train_token/1" | bc)

# default megatron-deepspeed confgiraution is 3000 million, but they train model using 300 billion tokens. we use 159 billion tokens, so we set 1.59 billion tokens to lr-warmup-tokens.
#lr_warmup_tokens_in_billion=1.59
lr_warmup_tokens_in_billion=2.06
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
#export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
export TIMER="timer.${MODELSIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_pytorch1.13"
CHECKPOINT_PATH=checkpoints/${MODELSIZE}_pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k/ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}
mkdir -p $CHECKPOINT_PATH
USE_CHECKPOINT=""
#USE_CHECKPOINT="--save $CHECKPOINT_PATH"
#USE_CHECKPOINT="--save $CHECKPOINT_PATH --load $CHECKPOINT_PATH"

# Set TOKENIZER
#(
VOCAB_FILE=$HOME/work/DeepSpeedFugaku/tokenizer/models/ver2/code${CODE_VOCAB_SIZE}k_en${EN_VOCAB_SIZE}k_ja${JA_VOCAB_SIZE}k.ver2.1.model
TOKENIZER="--tokenizer-type JapaneseSentencePiece --vocab-file $VOCAB_FILE"
#
#INPUT_PREFIX=dataset
#VOCAB_FILE=${INPUT_PREFIX}/gpt2-vocab.json
#MERGE_FILE=${INPUT_PREFIX}/gpt2-merges.txt
#TOKENIZER="--tokenizer-type GPT2BPETokenizer --vocab-file $VOCAB_FILE --merge-file $MERGE_FILE"

### setting dataset start
case ${TRAINDATA} in
  "codeparrot")
  ## codeparrot
  DATA_PATH=data/codeparrot/codeparrot_content_document
  ;;
  "wiki")
  ## wiki
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
  ;;
  *)
    echo "${TRAINDATA} is not supported"
    exit;;
esac
  ### setting dataset end

TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
DATA_PARALLEL_ARGS="--DDP-impl local"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"
#PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"
WANDB_ARGS=""
if [ $WANDBHEAD != "nouse" ]; then
  WANDB_ARGS="--wandb-name ${WANDBHEAD}-${MODELSIZE}-${TRAINDATA}-pp${PIPELINE_MODEL_PARALLEL_SIZE}_tp${TENSOR_MODEL_PARALLEL_SIZE}_dp${DATA_PARALLEL_SIZE}_gb${GLOBAL_BATCH}_GEMM${MYGEMM}"
fi

export OMP_NUM_THREADS=48

# train samples
seq_len=2048
# we use another termination condition, train_tokens, instead of train_samples.
# but not using train_samples causes error. so we set train_samples to a large number.
train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))

# Set TRAIN_ITERS
TRAIN_ITERS="--train-tokens $train_token --train-samples $train_samples --lr-decay-tokens $lr_decay_tokens --lr-warmup-tokens $lr_warmup_tokens"
if [ $SET_TRAIN_ITER != "0" ]; then
  TRAIN_ITERS="--train-iters $SET_TRAIN_ITER --lr-decay-iters 320000"
fi

export LD_PRELOAD=/local/fcc/inst/other/lib/libtcmalloc.so
#export OMP_WAIT_POLICY=ACTIVE
#if [ $RANK -eq 0 ]; then
#  echo "# DEBUG flag=" $A64FX_CBLAS_SGEMM_BATCH_DEGBUG
#  echo "# GLOBAL_BATCH=" $GLOBAL_BATCH
#  echo "# MYGEMM=" $MYGEMM
#fi

# https://www.fugaku.r-ccs.riken.jp/faq/20210428_01 , https://www.fugaku.r-ccs.riken.jp/bug/20210428_02
export UTOFU_SWAP_PROTECT=1

EchoAndRun() {
  if [ $RANK -eq 0 ]; then
    echo "command: $@"
  fi
  "$@"
}

#EchoAndRun numactl -m 4-7 -N 4-7 python pretrain_gpt_ptprof.py \

EchoAndRun numactl -m 4-7 -N 4-7 python pretrain_gpt.py \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads ${NUM_ATTENTION_HEADS} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --global-batch-size ${GLOBAL_BATCH} \
    --seq-length $seq_len \
    --max-position-embeddings $seq_len \
    ${TRAIN_ITERS} \
    ${USE_CHECKPOINT} \
    --data-path $DATA_PATH \
    ${TOKENIZER} \
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
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS \
    $WANDB_ARGS \
  --use-flush-denormal \
   --log-batch-size-to-tensorboard \
   --log-validation-ppl-to-tensorboard \
   --log-timers-to-tensorboard \
   --log-optimizer-states-to-tensorboard \
   --log-dir $LOGDIR \

    #--num-layers-per-virtual-pipeline-stage 1 \

    #--checkpoint-activations \
  #--use-timer \
    #--num-layers-per-virtual-pipeline-stage 2 \
    #--cpu-optimizer \
    #--cpu-torch-adam \
  #--wandb-name "30B-5iter-pp${PIPELINE_MODEL_PARALLEL_SIZE}-tp${TENSOR_MODEL_PARALLEL_SIZE}-dp${DATA_PARALLEL_SIZE}-gb${GLOBAL_BATCH}-GEMM${MYGEMM}-ja${JA_PERTCENT}_en${EN_PERTCENT}_code${CODE_PERTCENT}"

echo `date` ALL_DONE

