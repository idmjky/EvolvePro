#!/bin/bash
# Configuration values for SLURM job submission.
# One leading hash ahead of the word SBATCH is not a comment, but two are.
#SBATCH --time=24:00:00 
##SBATCH -x node[110]
#SBATCH --job-name=means
#SBATCH -n 1 
#SBATCH -N 1   
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=1  
#SBATCH --constraint=high-capacity    
#SBATCH --mem=200gb  

source ~/.bashrc
conda activate esm2

# here you can specify different fasta file names to generate embeddings for different proteins. 
study_names=("hc")

model_names=("esm2_t48_15B_UR50D")
fasta_path="./"
results_path="./results_means/"

repr_layers=48
toks_per_batch=512

for model_name in "${model_names[@]}"; do
  for study in "${study_names[@]}"; do
    command="python3 extract.py ${model_name} ${fasta_path}${study}.fasta ${results_path}${study}/${model_name} --toks_per_batch ${toks_per_batch} --include mean"
    echo "Running command: ${command}"
    eval "${command}"
  done
done
