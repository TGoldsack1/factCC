#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/5.3.0

source activate factcc


python3 data_generation/create_training_data.py 

