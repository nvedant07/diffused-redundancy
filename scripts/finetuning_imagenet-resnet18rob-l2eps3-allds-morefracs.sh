for seed in {1..5}
do
for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.02 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.03 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.04 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.075 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.15 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.25 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--pretrained True \
--batch_size 256 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.4 \
--seed $seed
done
done