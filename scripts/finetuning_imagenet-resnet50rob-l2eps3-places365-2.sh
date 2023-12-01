seed=2

# python -m partially_inverted_reps.finetuning \
# --source_dataset imagenet \
# --wandb_name partial_finetuning \
# --finetuning_dataset places365 \
# --finetune_mode linear \
# --model resnet50 \
# --batch_size 1024 \
# --step_lr 5 \
# --checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
# --append_path robustl2eps3 \
# --epochs 50 \
# --mode random \
# --fraction 0.004 \
# --pretrained True \
# --seed $seed

# python -m partially_inverted_reps.finetuning \
# --source_dataset imagenet \
# --wandb_name partial_finetuning \
# --finetuning_dataset places365 \
# --finetune_mode linear \
# --model resnet50 \
# --batch_size 1024 \
# --step_lr 5 \
# --checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
# --append_path robustl2eps3 \
# --epochs 50 \
# --mode random \
# --fraction 0.005 \
# --pretrained True \
# --seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--step_lr 5 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--fraction 0.01 \
--pretrained True \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--step_lr 5 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--fraction 0.05 \
--pretrained True \
--seed $seed