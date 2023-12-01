python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 1. \
--pretrained True \
--seed 2

for seed in {1}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset places365 \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.0005 \
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
--fraction 0.001 \
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
--fraction 0.002 \
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
--fraction 0.003 \
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
--fraction 0.004 \
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
--fraction 0.005 \
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
--fraction 0.1 \
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
--fraction 0.2 \
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
--fraction 0.3 \
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
--fraction 0.5 \
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
--fraction 0.8 \
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
--fraction 0.9 \
--pretrained True \
--seed $seed
done

