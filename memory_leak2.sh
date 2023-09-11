#! /bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=pt-Aug"
#PJM -L elapse=0:10:00
#PJM -L "node=2"
#PJM --mpi "proc=2"
#PJM --mpi "max-proc-per-node=1"
#PJM -g hp190122
#PJM -x PJM_LLIO_GFSCACHE=/vol0003:/vol0004
#PJM -j
#PJM -S

# # if fugaku environment
# if [ "$USER" == "u01959" ]; then

# # notice start job
# export PATH=$PATH:/data/hp190122/users/u01959/jobscheduler2slack
# post_message start_job

# # activate pytorch environment
# source /data/hp190122/share/PyTorch-1.10.1/env.src
# export PYTHONUSERBASE=$HOME/work/.local
# export PATH=$PATH:$PYTHONUSERBASE/bin
# fi
# # setting distribute
# MASTER_ADDR=localhost
# MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
# WORLD_SIZE=2
# output_path="jobs/torch.1.10/outs/${PJM_JOBID}_n${nodos}"

# # run
# if [ "$USER" == "u01959" ]; then
# MASTER_ADDR=$MASTER_ADDR \
# MASTER_PORT=$MASTER_PORT \
# mpiexec -n ${WORLD_SIZE} -std-proc ${output_path}/stdproc \
# -x LD_PRELOAD="/vol0003/hp190122/data/users/u01959/my_mpi_allreduce/only_override.so" \
# python memory_leak.py
# else
# MASTER_ADDR=$MASTER_ADDR \
# MASTER_PORT=$MASTER_PORT \
# mpiexec -n ${WORLD_SIZE} \
# python memory_leak2.py
# fi

export PATH=$PATH:/data/hp190122/users/u01959/jobscheduler2slack
post_message start_job
source /data/hp190122/share/PyTorch-1.10.1/env.src
export PYTHONUSERBASE=$HOME/work/.local
export PATH=$PATH:$PYTHONUSERBASE/bin
MASTER_ADDR=localhost
MASTER_PORT=$((10000 + ($PJM_JOBID % 50000)))
WORLD_SIZE=2
output_path="jobs/torch.1.10/outs/${PJM_JOBID}_n${nodos}"
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
mpiexec -n ${WORLD_SIZE} -std-proc ${output_path}/stdproc \
-x LD_PRELOAD="/vol0003/hp190122/data/users/u01959/my_mpi_allreduce/only_override.so" \
python memory_leak2.py
