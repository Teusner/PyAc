#!/bin/bash

#SBATCH --job-name=Proteus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --time=5-00:00:00 
#SBATCH --mem=10g

#SBATCH --output="%x.out%j"     # fichier de sortie ${SLURM_JOB_NAME}.out${SLURM_JOB_ID}
#SBATCH --comment="Running Proteus FDTD simulation"

# Load software
module load anaconda/3-2020.07
conda activate /home/brateaqu/PyAc/proteus

python3 src/Proteus.py