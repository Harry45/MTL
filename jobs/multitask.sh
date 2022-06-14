#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=00:30:00
#SBATCH --job-name=multitask
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=short
#SBATCH --cluster=htc
#SBATCH --gres=gpu:v100:1
#SBATCH --output=jobs/%j.out

echo Multitask Weighted Loss

nvidia-smi

module load Anaconda3
export CONPREFIX=/data/phys-zooniverse/phys2286/pytorch-env39
source activate $CONPREFIX
echo $CONPREFIX

which python

echo Training started.
date "+%H:%M:%S   %d/%m/%y"

time python train_MTL.py

echo Training completed.
date "+%H:%M:%S   %d/%m/%y"