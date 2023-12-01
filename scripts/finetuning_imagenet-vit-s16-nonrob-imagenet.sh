for seed in {1..5}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.001 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset imagenet \
--finetune_mode linear \
--model vit_small_patch16_224 \
--batch_size 1024 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/S_16-i1k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob \
--epochs 25 \
--mode random \
--step_lr 5 \
--fraction 0.9 \
--seed $seed
done
