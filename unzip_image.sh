#!/bin/bash

#SBATCH -J unzip
#SBATCH -p shared
#SBATCH -N 1                    # number of nodes
#SBATCH -n 16                   # number of cores
#SBATCH --gres=gpu:0
#SBATCH --mem 64000              # memory pool for all cores
#SBATCH -t 7-00:00              # time (D-HH:MM)
#SBATCH --export=ALL
#SBATCH -o Job.%N.%j.out # STDOUT
#SBATCH -e Job.%N.%j.err # STDERR

unzip /n/holyscratch01/mickley/dpendergrass/band1_2018_png.zip -d /n/mickley/lab/HMS_vision/band1_2018
