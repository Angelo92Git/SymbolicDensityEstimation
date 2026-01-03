#!/bin/bash
#SBATCH --job-name=dijet_nde                        # Job name
#SBATCH --time=0-16:00:00                           # Time limit hrs:min:sec
#SBATCH --ntasks=1                                  # Number of tasks (processes)
#SBATCH --cpus-per-task=25                          # Number of CPU cores per task (maximum on compute canada is 50)
#SBATCH --mem=128G                                  # Memory per node
#SBATCH --output=logs/slurm-%j-%N_dijet_nde.out     # the print of xxx.jl will be logged in this file, %N for node name, %j for job id:w

module load StdEnv/2023 julia/1.11.3
julia --thread=64 sdes_pipeline.jl "dijet_neural" "config_dijet" "false"

# module load python/3.12
# module load scipy-stack
# pip install KDEpy
# pip install --upgrade KDEpy
# srun --cpus-per-task=1 --mem=25G --time=01:00:00 --pty bash
## on fir
# srun --partition=cpupreempt --cpus-per-task=1 --mem=32G --time=01:00:00 --pty bash
# scp aa3rajen@fir.alliance.ca:projects/def-vganesh/aa3rajen/SymbolicDensityEstimation/data/processed_data/gaussian_4d_joint_data.zip ./data/processed_data 
