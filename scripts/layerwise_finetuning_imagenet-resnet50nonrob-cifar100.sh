for layer in {0..16}
do
python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 1. \
--seed 2 \
--layers $layer
done

for seed in {1..5}
do
for layer in {0..16}
do
python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.0005 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.005 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.05 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.1 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.2 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.4 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.5 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.8 \
--seed $seed \
--layers $layer

python -m partially_inverted_reps.finetuning_on_middle_layers \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--append_path nonrob \
--epochs 50 \
--step_lr 10 \
--save_every 0 \
--mode random \
--fraction 0.9 \
--seed $seed \
--layers $layer
done
done
