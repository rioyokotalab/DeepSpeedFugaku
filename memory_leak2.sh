#! /bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=0:10:00
#PJM -L "node=2"
#PJM --mpi "proc=2"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

# load pytorch
source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin

# set up environment
MASTER_ADDR=localhost
MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
WORLD_SIZE=2
output_path="jobs/torch.1.10/outs/${PJM_JOBID}_n${nodos}"

# run
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
mpiexec -n ${WORLD_SIZE} -std-proc ${output_path}/stdproc \
python memory_leak2.py
