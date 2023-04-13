#!/bin/bash
#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB


source /home/${USER}/.bashrc
srun mpiexec ./main model.bin output.txt 1 1203 $@