#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=125:00:00
#SBATCH --job-name=multilabel
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=long
#SBATCH --clusters=all
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jobs/%j.out

echo Starting Multilabel Weighted

nvidia-smi

module purge
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

echo Training started.
date "+%H:%M:%S   %d/%m/%y"

time python train_ML.py

echo Training completed.
date "+%H:%M:%S   %d/%m/%y"