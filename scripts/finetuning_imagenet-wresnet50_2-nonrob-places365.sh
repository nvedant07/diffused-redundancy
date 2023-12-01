python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 1. \
--seed 2

for seed in {1..5}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.001 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.002 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.003 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.004 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model wide_resnet50_2 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.9 \
--seed $seed
done