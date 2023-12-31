python -m partially_inverted_reps.finetuning \
--source_dataset cifar100 \
--finetuning_dataset cifar100 \
--finetune_mode full \
--base_dir partially_inverted_reps \
--save_every 0 \
--model resnet50 \
--pretrained False \
--batch_size 512 \
--append_path scratch \
--epochs 100 \
--optimizer sgd \
--lr 0.001 \
--step_lr 500 \
--warmup_steps 100 \
--gradient_clipping 1.0
