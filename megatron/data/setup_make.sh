#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=10:00:00
#PJM -L "node=1"
#PJM --llio localtmp-size=80Gi
#PJM --mpi "proc=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

mkdir -p /local/fcc/pytorch
cd /local/fcc
tar xf /home/u11890/work/1693389241.318550480.fcc.pytorch.y.r1.13_for_a64fx.tar
source /local/fcc/inst/venv/bin/activate
cd /home/u11890/work/timer/DeepSpeedFugaku/megatron/data
make

