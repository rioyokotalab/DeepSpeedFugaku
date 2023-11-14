#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=1:00:00
#PJM -L "node=24x12x32:torus:strict-io"
#PJM -L "freq=2200"
#PJM --llio localtmp-size=70Gi
#PJM --llio sharedtmp-size=10Gi
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004:/vol0005:/vol0006
#PJM -j
#PJM -S

## These definitions will be inserted by submit command
#DEEPSPEEDFUGAKU=<path-to>/DeepSpeedFugaku
#PYTORCH_TGZ=<path>/...fcc.pytorch...tar.gz
#JOBSCRIPT=<path-to-script>/job.sh
#MODEL="30b"
#TRAINDATA="codeparrot"
#WANDBHEAD="nouse"
#LOGDIR=log/stdout
#TRAIN_ITERS=0

PP=24  # pipeline-parallel
TP=24  # tensor-parallel
DP=16 # data-parallel
GB=1536 # global-batch
MB=1  # micro-batch
MYGEMM=99 # 1: original, 99: fj-BMM-v1
GEMM_DEBUG=0 # 1: output shapes to calculate by fj-BMM

NNODES=$(( $PP * $TP * $DP ))

EchoAndRun() {
  echo "command: $@"
  "$@"
}

SDIR=`readlink -f "$0" | xargs dirname`

llio_transfer $PYTORCH_TGZ
llio_transfer $JOBSCRIPT
llio_transfer ${DEEPSPEEDFUGAKU}/pretrain_gpt.py
dir_transfer ${DEEPSPEEDFUGAKU}/megatron

touch ./JOBID_$PJM_JOBID
EchoAndRun mpirun -np $NNODES -std-proc ${LOGDIR} \
         $JOBSCRIPT $DEEPSPEEDFUGAKU $PYTORCH_TGZ $MODEL $TRAINDATA $PP $TP $DP $GB $MB $MYGEMM $GEMM_DEBUG $WANDBHEAD $TRAIN_ITERS

