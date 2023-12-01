# python -m partially_inverted_reps.finetuning \
# --source_dataset imagenet21k \
# --finetuning_dataset oxford-iiit-pets \
# --finetune_mode linear \
# --model vit_small_patch32_224 \
# --batch_size 256 \
# --append_path nonrob \
# --epochs 100 \
# --mode random \
# --step_lr 20 \
# --save_every 0 \
# --fraction 1. \
# --seed 2 \
# --checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz


for seed in {5..10}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.005 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.05 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.1 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.2 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.5 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet21k \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model vit_small_patch32_224 \
--batch_size 256 \
--append_path nonrob \
--epochs 100 \
--mode random \
--step_lr 20 \
--save_every 0 \
--fraction 0.9 \
--seed $seed \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz
done