#!/bin/bash

: ${NODES:=1}
salloc -N $NODES --partition P1 --exclusive --gres=gpu:1 \
    mpirun --bind-to none -mca btl ^openib -npernode 1 \
    numactl --physcpubind 0-63 \
    ./main $@
