#!/bin/bash
#
#SBATCH -p a100
#SBATCH --gres=gpu:2
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     
#SBATCH -a 5
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written


srun --jobid $SLURM_JOBID bash -c 'python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--append_path nonrob \
--epochs 50 \
--mode random \
--pretrained True \
--step_lr 5 \
--fraction 0.004 \
--seed ${SLURM_ARRAY_TASK_ID}'

srun --jobid $SLURM_JOBID bash -c 'python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--append_path nonrob \
--epochs 50 \
--mode random \
--pretrained True \
--step_lr 5 \
--fraction 0.005 \
--seed ${SLURM_ARRAY_TASK_ID}'

srun --jobid $SLURM_JOBID bash -c 'python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--append_path nonrob \
--epochs 50 \
--mode random \
--pretrained True \
--step_lr 5 \
--fraction 0.01 \
--seed ${SLURM_ARRAY_TASK_ID}'

srun --jobid $SLURM_JOBID bash -c 'python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--append_path nonrob \
--epochs 50 \
--mode random \
--pretrained True \
--step_lr 5 \
--fraction 0.05 \
--seed ${SLURM_ARRAY_TASK_ID}'

