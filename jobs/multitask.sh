#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=125:00:00
#SBATCH --job-name=multitask
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=long
#SBATCH --cluster=htc
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jobs/%j.out

echo Checking GPU
nvidia-smi

echo Checking Unloading modules
module purge

echo Loading modules
module load CUDA/11.3.1
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

echo Training started.
date "+%H:%M:%S   %d/%m/%y"

time python train_MTL.py

echo Training completed.
date "+%H:%M:%S   %d/%m/%y"