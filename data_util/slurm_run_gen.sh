#!/bin/bash
#SBATCH --job-name=gen_train_tfrecords
#SBATCH --nodes=1
#SBATCH --cpus=16
#SBATCH --mem=64GB
#SBATCH --time=36:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --output=/usr/stud/wangyu/PycharmProjects/slurm_log/slurm-%j.out
export PYTHONHOME=/usr/stud/wangyu/venv
srun /usr/stud/wangyu/venv/bin/python gen_bbox_track_train_tfrecord.py
