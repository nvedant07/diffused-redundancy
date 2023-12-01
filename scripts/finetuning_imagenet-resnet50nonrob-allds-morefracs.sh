for seed in {1..5}
do
for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.02 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.03 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.04 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.075 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.15 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.25 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--pretrained True \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.4 \
--seed $seed
done
done