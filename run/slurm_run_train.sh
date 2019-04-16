#!/bin/bash
#SBATCH --job-name=bbox_mask_prop_gpu4_b128
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128GB
#SBATCH --gres=gpu:p6000:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=4-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50, TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out

export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python train.py
