python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
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
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.0005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.001 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.005 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.01 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.05 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.1 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.2 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.3 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.5 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.8 \
--seed $seed

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset places365 \
--finetune_mode linear \
--model vgg16_bn \
--batch_size 512 \
--checkpoint_path /NS/robustness_3/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt \
--append_path robustl2eps3 \
--epochs 50 \
--mode random \
--step_lr 10 \
--fraction 0.9 \
--seed $seed
done