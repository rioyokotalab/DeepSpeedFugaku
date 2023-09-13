#include <mpi.h>
#include <stdlib.h>
#include <string.h>

//int system(const char*);

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    system("echo before call mpi by cpp max: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.max_usage_in_bytes");
    system("echo before call mpi by cpp cur: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.usage_in_bytes");
    int ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    system("echo after  call mpi by cpp max: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.max_usage_in_bytes");
    system("echo after  call mpi by cpp cur: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.usage_in_bytes");
    return ret;
}
