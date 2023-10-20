#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=ppu2023"
#PJM -L elapse=10:00:00
#PJM -L "node=8x48x32:torus:strict-io"
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

hostfile_name='4x16x16x2x3x2_tp6dp256pp8'
stdproc_name='jobs/30b_pp8_tp6_dp256_pytorch1.13/outs/${PJM_JOBID}_n/stdproc'
inner_file_name='30b_pp8_tp6_dp256_rankmap_inner.sh'
num_node=12288

rm /home/u11890/work/rankmap/vcoordfile_${hostfile_name}

llio_transfer /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

mpirun -n ${num_node} /home/u11890/work/rankmap/fjmpi_6d_to_3d.out /home/u11890/work/rankmap/hostfile_${hostfile_name} /home/u11890/work/rankmap/vcoordfile_${hostfile_name}

llio_transfer --purge /home/u11890/work/rankmap/fjmpi_6d_to_3d.out

llio_transfer ${inner_file_name}
llio_transfer /home/u11890/work/1693389241.318550480.fcc.pytorch.y.r1.13_for_a64fx.tar

mpirun -n ${num_node} \
	--vcoordfile /home/u11890/work/rankmap/vcoordfile_${hostfile_name} \
	-std-proc ${stdproc_name} \
	bash ${inner_file_name}
