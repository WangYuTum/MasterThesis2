#!/bin/bash
#SBATCH --job-name=multi_gpu_inf
#SBATCH --nodes=1
#SBATCH --cpus=8
#SBATCH --mem=16GB
#SBATCH --gres=gpu:2
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=10:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out
export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python val_imagenet.py
