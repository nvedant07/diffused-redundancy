python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 1 \
--seed 2

for seed in {1..5}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar10 \
--finetune_mode linear \
--model wide_resnet50_4 \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/wideresnet-50-4-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--step_lr 10 \
--mode random \
--fraction 0.9 \
--seed $seed
done