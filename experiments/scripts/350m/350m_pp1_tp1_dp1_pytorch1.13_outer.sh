#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small"
#PJM -L elapse=10:00:00
#PJM -L "node=1"
#PJM --llio localtmp-size=80Gi
#PJM --mpi "proc=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

mpirun -np 1 -std-proc "jobs/350m_pp1_tp1_dp1_pytorch1.13/outs/${PJM_JOBID}_n/stdproc" \
	./350m_pp1_tp1_dp1_pytorch1.13_inner.sh
