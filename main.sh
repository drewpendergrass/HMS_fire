#!/bin/bash

#SBATCH -J H_v_t
#SBATCH -p gpu
#SBATCH -N 1                    # number of nodes
#SBATCH -n 8                    # number of cores
#SBATCH --gres=gpu:1
#SBATCH --mem 256000              # memory pool for all cores
#SBATCH -t 6-00:00              # time (D-HH:MM)

module load Anaconda3/5.0.1-fasrc01 cuda/10.1.243-fasrc01 cudnn/7.6.5.32_cuda10.1-fasrc01
source activate tf-gpu #&& cd ~/HMS_vision
python main.py --side-len=256 --batch-size=32 --seed=107 > log_train_UNBALANCE_jaccard_$(date "+%Y.%m.%d-%H.%M.%S").txt 2>&1
