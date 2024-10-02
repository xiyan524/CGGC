#!/bin/bash
#
#SBATCH --job-name=py
#SBATCH --output=logs/%j.log
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --partition=afkm
#SBATCH --gres=gpu:mem24g:2

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

max_batch_size=$1
icl_num=$2
model_name=$3
demonstration_type=$4

echo "max_batch_size: $max_batch_size"
echo "icl_num: $icl_num"
echo "model_name: $model_name"
echo "demonstration_type: $demonstration_type"

# JOB STEPS 
source xxx
#nvidia-smi
nvidia-debugdump --list
folder_path=result/$model_name
if [ ! -d "$folder_path" ]; then
  mkdir $folder_path
fi

python3 -u main.py --data_path xxx --output_path xxx --icl_num $icl_num --max_batch_size $max_batch_size --max_seq_gen 20 --model_name $model_name --prompt_str "generate a natural sentence with the provided concepts and their commonsense reasoning graphs:\n " --demonstration_type $demonstration_type --prompt_type cps_graphs_2_cps --random_seed 106522
