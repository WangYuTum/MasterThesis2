#!/bin/bash
#SBATCH --job-name=aug_pascal12
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=4-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90

export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python gen_train_pairs.py
