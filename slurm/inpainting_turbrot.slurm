#!/bin/bash
#SBATCH -p std
#SBATCH --time 24:00:00      # format: HH:MM:SS
##SBATCH -w a100-2           # ask for specific host
#SBATCH -N 1                 # xxx node
#SBATCH --ntasks-per-node=16 # xxx core per node
#SBATCH --gres=gpu:a100:4    # xxx gpus per node
#SBATCH --mem=256000         # memory per node out of xxx MB
#SBATCH --job-name=turbrot
##SBATCH --mail-type=ALL
##SBATCH --mail-user=<user_email>

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate Palette-Lagr

#python run.py -p train -c config/inpainting_turbrot.json -gpu 0,1,2,3
python run.py -p test -c config/inpainting_turbrot.json -gpu 0,1,2,3
