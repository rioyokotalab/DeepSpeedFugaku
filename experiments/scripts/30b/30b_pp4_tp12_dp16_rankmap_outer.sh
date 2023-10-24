#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=10:00:00
#PJM -L "node=8x12x8:torus:strict-io"
#PJM -L "freq=2200"
#PJM -L "throttling_state=0"
#PJM -L "issue_state=0"
#PJM -L "ex_pipe_state=0"
#PJM -L "eco_state=0"
#PJM -L "retention_state=0"
#PJM --llio localtmp-size=70Gi
#PJM --llio sharedtmp-size=10Gi
#PJM --mpi "proc=1"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp230254
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004:/vol0005:/vol0006
#PJM -j
#PJM -S

pp=4
tp=12
dp=16
gbs=96
num_node=768
hostfile_name="4x4x4x2x3x2_tp${tp}dp${dp}pp${pp}"
param_name="30b_pp${pp}_tp${tp}_dp${dp}_pytorch1.13_rankmap_gbs${gbs}"
stdproc_name="jobs/${param_name}/outs/${PJM_JOBID}_n/stdproc"

rm /home/u11890/work/rankmap/vcoordfile_${hostfile_name}

llio_transfer /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

mpirun -n ${num_node} /home/u11890/work/rankmap/fjmpi_6d_to_3d.out /home/u11890/work/rankmap/hostfile_${hostfile_name} /home/u11890/work/rankmap/vcoordfile_${hostfile_name}

llio_transfer --purge /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

llio_transfer inner.sh
llio_transfer /home/u11890/work/1693389241.318550480.fcc.pytorch.y.r1.13_for_a64fx.tar

mpirun -n ${num_node} \
	--vcoordfile /home/u11890/work/rankmap/vcoordfile_${hostfile_name} \
	-std-proc ${stdproc_name} \
	bash inner.sh ${param_name} ${num_node} ${pp} ${tp} ${dp} ${gbs}

