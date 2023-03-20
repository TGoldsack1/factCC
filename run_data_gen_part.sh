#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --mem=160GB
#SBATCH --nodes=1
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/5.3.0

source activate factcc

export GOOGLE_APPLICATION_CREDENTIALS="/home/acp20tg/factCC/data_generation/google_key.json"

python3 data_generation/combine_negative_files.py
python3 data_generation/create_data_part.py data_generation/raw_data/st1_PLOS_formatted.jsonl --all_augmentations --save_intermediate

#python3 data_generation/create_data.py data_generation/raw_data/st1_eLife_formatted.jsonl --all_augmentations --save_intermediate
