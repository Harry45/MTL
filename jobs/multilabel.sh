#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=00:30:00
#SBATCH --job-name=multitask
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=short
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/%j.out

nvidia-smi

module purge
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

echo Training started.
date "+%H:%M:%S   %d/%m/%y"

python train_ML.py

echo Training completed.
date "+%H:%M:%S   %d/%m/%y"