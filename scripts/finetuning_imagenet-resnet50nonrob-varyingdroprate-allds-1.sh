# for ft_ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
for ft_ds in {"flowers","oxford-iiit-pets"}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--wandb_name partial_finetuning \
--finetuning_dataset $ft_ds \
--finetune_mode linear \
--model resnet50 \
--batch_size 1024 \
--config_file partially_inverted_reps/configs/resnet50_drop_rate_01.yaml \
--epochs 20 \
--mode random \
--pretrained True \
--step_lr 3 \
--fraction 1. \
--seed 2
done