#!/bin/bash
#SBATCH --job-name=dijet_data                            # Job name
#SBATCH --partition=gpubase_bygpu_b2                  # interactive GPU
#SBATCH --time=4:00:00                                # Time limit hrs:min:sec
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                                    # Number of tasks (processes)
#SBATCH --cpus-per-task=4                             # Number of CPU cores per task (maximum on compute canada is 50)
#SBATCH --mem=20G                                     # Memory per node
#SBATCH --output=logs/slurm-%j-%N_dijet_data.out         # the print of xxx.jl will be logged in this file, %N for node name, %j for job id:w

module load python/3.12
module load scipy-stack
source ~/torch_env/bin/activate
python -m data_processing_scripts.gen_data data_config_dijet