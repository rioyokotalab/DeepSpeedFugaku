#PJM -L "node=2"
#PJM --mpi "proc=2"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

# prepare for environment of pytorch-1.10
source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

# Change for multinode config
CPUS_PER_NODE=1
NNODES=2
NODE_RANK=0
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))
export MASTER_ADDR=localhost
export MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
PP=1
TP=1
DP=$(($WORLD_SIZE / ($TP * $PP)))
TORCH_VERSION=1.10
MODEL_SIZE=1.3b
export TIMER="timer/torch${TORCH_VERSION}_${MODEL_SIZE}_${PP}p_${TP}t_${DP}d/${PJM_JOBID}"
DATE=`date +%Y%m%d-%H%M%S`
CHECKPOINT_PATH=checkpoints/torch${TORCH_VERSION}_${MODEL_SIZE}_pp${PP}_mp${TP}_dp${DP}/${DATE}/
INPUT_PREFIX=/vol0003/hp190122/data/users/u01959/data
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=${INPUT_PREFIX}/BookCorpusDataset_text_document
TENSORBOARD_ARGS="--tensorboard-dir experiments/tensorboard"

output_path="jobs/${PJM_JOBID}_n${nodos}"
DISTRIBUTED_ARGS="-np $NNODES -std-proc ${output_path}/stdproc --mca coll_select_allreduce_algorithm ring"
DATA_PARALLEL_SIZE=${DP}
PIPELINE_MODEL_PARALLEL_SIZE=${PP}
TENSOR_MODEL_PARALLEL_SIZE=${TP}
PIPELINE_PARALLEL_ARGS="--pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE"
MODEL_PARALLEL_ARGS="--tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE"
#DATA_PARALLEL_ARGS="--DDP-impl torch"
#PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $DATA_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"
PARALLEL_ARGS="$MODEL_PARALLEL_ARGS $PIPELINE_PARALLEL_ARGS"

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(($DATA_PARALLEL_SIZE*$MICRO_BATCH_SIZE))
export OMP_NUM_THREADS=48

mpirun $DISTRIBUTED_ARGS \
-x LD_PRELOAD="./override/only_override.so" \
python pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --train-iters 300 \
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
    --eval-iters 1 \
    --no-cuda \
    --use-cpu-initialization \
    --num-workers 0 \
    --no-load-rng \
    $PARALLEL_ARGS \
    $TENSORBOARD_ARGS

