for ds in {"cifar10","cifar100","flowers","oxford-iiit-pets"}
do

python -m partially_inverted_reps.calculate_pca \
--source_dataset imagenet \
--finetuning_dataset $ds \
--model vgg16_bn \
--batch_size 512 \
--append_path nonrob

done