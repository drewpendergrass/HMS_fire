#!/bin/bash

#SBATCH -J HMS_split
#SBATCH -p huce_intel
#SBATCH -N 1                    # number of nodes
#SBATCH -n 16                   # number of cores
#SBATCH --mem 100000              # memory pool for all cores
#SBATCH -t 1-00:00              # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o Job.%N.%j.out # STDOUT
#SBATCH -e Job.%N.%j.err # STDERR

module load Anaconda3/5.0.1-fasrc01
#source activate tf-gpu && cd ~/HMS_vision/
python produce_tiles.py > logs/log_split_$(date "+%Y.%m.%d-%H.%M.%S").txt 2>&1
