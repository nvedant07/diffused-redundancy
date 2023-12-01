python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 1. \
--seed 2

for seed in {1..5}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.001 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.002 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.003 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.004 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset flowers \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--step_lr 20 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--mode random \
--fraction 0.9 \
--seed $seed
done

