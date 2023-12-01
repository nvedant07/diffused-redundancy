for ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do

python -m partially_inverted_reps.calculate_pca \
--source_dataset imagenet21k \
--finetuning_dataset $ds \
--model vit_small_patch16_224 \
--batch_size 512 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet21k/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz \
--append_path nonrob

done