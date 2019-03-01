#!/bin/bash
#SBATCH --job-name=insane_fast_train_gpu4_b64
#SBATCH --nodes=1
#SBATCH --cpus=16
#SBATCH --mem=96GB
#SBATCH --gres=gpu:titanxpascal:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_50, TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out

export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python train.py
