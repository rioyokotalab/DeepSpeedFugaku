#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM --rsc-list "proc-openfd=65536"
#PJM -L elapse=24:00:00
#PJM -L "freq=2200"
#PJM -L "node=1024"
#PJM --mpi "proc=1024"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM --llio localtmp-size=80Gi
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004:/vol0005:/vol0006
#PJM -j
#PJM -S

PP=4   # pipeline-parallel
TP=1   # tensor-parallel
DP=256 # data-parallel
MB=1   # micro-batch
GB=512 # global-batch
AHEADS=16 # attention heads
MYGEMM=99 # 1: original, 99: fj-BMM-v1
GEMM_DEBUG=0 # 1: output shapes to calculate by fj-BMM

CPUS_PER_NODE=1
NNODES=$(( $PP * $TP * $DP ))
export WORLD_SIZE=$(($CPUS_PER_NODE*$NNODES))

JOB_SCRIPT_DIR=${HOME}/work/DeepSpeedFugaku/experiments/scripts/1.3b-fj

mpirun -np $WORLD_SIZE -std-proc "jobs/1.3b_8215tgz/outs/${PJM_JOBID}_n/stdproc" \
        ${JOB_SCRIPT_DIR}/job_1.3b_loss.sh $PP $TP $DP $GB $MYGEMM $MB $AHEADS $GEMM_DEBUG

