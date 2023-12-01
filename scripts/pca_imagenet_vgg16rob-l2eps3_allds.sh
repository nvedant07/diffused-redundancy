for ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do

python -m partially_inverted_reps.calculate_pca \
--source_dataset imagenet \
--finetuning_dataset $ds \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3

done