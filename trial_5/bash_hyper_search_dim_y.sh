#!/bin/bash
#SBATCH -A ACCOUNT_name
#SBATCH --partition partition_name     
#SBATCH -N 1
#SBATCH -n 24
#SBATCH	--mail-user=email@gmail.com
#SBATCH	--mail-type=END,FAIL
#SBATCH --time 72:00:00
#SBATCH --job-name dim_y
#SBATCH --output dim_y-%J.log

singularity exec jupyter.mictlan.10.v5.sif python3 keras_tuner_tensorflow_projection_umap_dim_y.py

