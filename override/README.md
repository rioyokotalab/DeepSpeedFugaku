# override

MPI_Allreduceをオーバーライドして自作のallreduceをpython側から呼び出す

- only_override.c
    - MPI_Allreduceを乗っ取って，PMPI_Allreduceを呼び出すだけ
    - 呼び出しの前後でメモリ使用量を計測している

