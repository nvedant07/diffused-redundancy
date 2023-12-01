#!/bin/bash
#
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     
#SBATCH -a 1-4
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written

srun --jobid $SLURM_JOBID bash -c './partially_inverted_reps/scripts/layerwise_finetuning_imagenet-resnet50rob-l2eps3-allds-${SLURM_ARRAY_TASK_ID}.sh'