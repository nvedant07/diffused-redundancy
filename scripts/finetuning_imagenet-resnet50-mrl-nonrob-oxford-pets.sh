for num_features in {8,16,32,64,128,256,512,1024}
do
python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50_mrl \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/r50_mrl1_e0_ff2048.pt \
--append_path nonrob \
--epochs 50 \
--mode first \
--step_lr 10 \
--num_features $num_features \
--seed 1
done

python -m partially_inverted_reps.finetuning \
--source_dataset imagenet \
--finetuning_dataset oxford-iiit-pets \
--finetune_mode linear \
--model resnet50_mrl \
--batch_size 256 \
--checkpoint_path /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/imagenet1k/r50_mrl1_e0_ff2048.pt \
--append_path nonrob \
--epochs 50 \
--mode first \
--step_lr 10 \
--num_features 2048 \
--seed 2