#!/bin/bash
#SBATCH --job-name=train_gpu1_b128_sgd
#SBATCH --nodes=1
#SBATCH --cpus=6
#SBATCH --mem=24GB
#SBATCH --gres=gpu:p6000:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50, TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out
export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python train_imagenet.py
