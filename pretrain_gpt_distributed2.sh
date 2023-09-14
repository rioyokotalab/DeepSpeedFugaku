#PJM -L "node=2"
#PJM --mpi "proc=2"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

# Runs the "345M" parameter model

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATE=`date "+%Y%m%d_%H%M%S"`
CHECKPOINT_PATH=checkpoints/gpt2_$DATE
INPUT_PREFIX=/vol0003/hp190122/data/users/u01959/data
DATA_PATH=${INPUT_PREFIX}/BookCorpusDataset_text_document
# DATA_PATH=/mnt/nfs/Users/nakamura459/wiki_ja_text_document

#DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
DISTRIBUTED_ARGS="-n $GPUS_PER_NODE"


MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
WORLD_SIZE=$WORLD_SIZE \
mpiexec $DISTRIBUTED_ARGS \
-x LD_PRELOAD="override/only_override.so" \
       python pretrain_gpt2.py \
       --micro-batch-size 8 \
       --global-batch-size 64 \
       --train-iters 2 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ${INPUT_PREFIX}/gpt2-vocab.json \
       --merge-file ${INPUT_PREFIX}/gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend mpi \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --deepspeed_mpi
