#!/bin/bash

# Example of running python script with a job array

#SBATCH -J Proteus
#SBATCH --array=0-100                  # how many tasks in the array
#SBATCH -c 1                            # one CPU core per task
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=quentin.brateau@ensta-bretagne.org

# Load software
module load anaconda/3-2020.07

conda activate /home/brateaqu/PyAc/proteus

# Run python script with a command line argument
srun python3 src/Proteus_image.py $SLURM_ARRAY_TASK_ID
