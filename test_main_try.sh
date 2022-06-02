#!/bin/bash

#SBATCH -J H_v_t
#SBATCH -p gpu_test
#SBATCH -N 1                    # number of nodes
#SBATCH -n 8                    # number of cores
#SBATCH --gres=gpu:1
#SBATCH --mem 256000              # memory pool for all cores
#SBATCH -t 0-07:50              # time (D-HH:MM)
#SBATCH --export=ALL

#module load Anaconda3/5.0.1-fasrc01 cuda/10.1.243-fasrc01 cudnn/7.6.5.32_cuda10.1-fasrc01
python test_training.py > log_train_$(date "+%Y.%m.%d-%H.%M.%S").txt 2>&1
