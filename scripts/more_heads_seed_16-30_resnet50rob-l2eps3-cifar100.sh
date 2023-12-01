for seed in {20..30}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--step_lr 20 \
--mode random \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--step_lr 20 \
--mode random \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset cifar100 \
--finetune_mode linear \
--model resnet50 \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt \
--append_path robustl2eps3 \
--epochs 100 \
--step_lr 20 \
--mode random \
--fraction 0.3 \
--seed $seed
done

