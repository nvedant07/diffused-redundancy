for ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do

python -m partially_inverted_reps.calculate_pca \
--source_dataset imagenet \
--finetuning_dataset $ds \
--model wide_resnet50_2 \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps0/wideresnet-50-2-nonrob.ckpt \
--append_path nonrob

done