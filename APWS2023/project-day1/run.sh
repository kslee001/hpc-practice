#!/bin/bash

srun -N 1 --partition apws --exclusive --job-name=unet ./main $@
