for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 1. \
--seed 2
done

for seed in {1..5}
do
for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet18 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.9 \
--seed $seed
done
done