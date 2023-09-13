#include <mpi.h>
#include <stdlib.h>
#include <string.h>

//int system(const char*);

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    system("echo before call mpi by cpp max: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.max_usage_in_bytes");
    system("echo before call mpi by cpp cur: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.usage_in_bytes");
    int ret;
    /*if(sendbuf==MPI_IN_PLACE) {
        int type_size;
        MPI_Type_size(datatype, &type_size);
        void *tmpbuf = malloc((long)type_size * count);
        memcpy(tmpbuf, recvbuf, (long)type_size * count);
        ret = PMPI_Allreduce(tmpbuf, recvbuf, count, datatype, op, comm);
        free(tmpbuf);
        //return ret;
    } else {*/
        ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    //}
    system("echo after  call mpi by cpp max: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.max_usage_in_bytes");
    system("echo after  call mpi by cpp cur: ;cg=`grep memory /proc/self/cgroup`;cat /sys/fs/cgroup/memory/${cg#*memory:/}/memory.usage_in_bytes");
    return ret;
}
