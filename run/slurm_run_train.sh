#!/bin/bash
#SBATCH --job-name=train_gpu4_b512
#SBATCH --nodes=1
#SBATCH --cpus=16
#SBATCH --mem=80GB
#SBATCH --gres=gpu:titanxpascal:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50, TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out

export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python train.py
