#!/bin/bash
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=arrykrish@gmail.com
#SBATCH --time=23:30:00
#SBATCH --job-name=multi-label-network
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=128G
#SBATCH --partition=long
#SBATCH --cluster=htc
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/%j.out

module purge
module load Anaconda3
export CONPREFIX=$DATA/pytorch-env39
source activate $CONPREFIX

echo Training started.

python train_ML.py

echo Training completed.

echo Performing testing.

python prediction_ML.py

echo Prediction completed.

rm -r __pycache__/