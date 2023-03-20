#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --mem=128GB
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/5.3.0
#module load CUDA/10.1.243
module load CUDAcore/11.1.1

source activate factcc

wandb login ccda32c4c849948d438ffe0976e2f783833656b0

bash ./modeling/scripts/my-factcc-finetune.sh
