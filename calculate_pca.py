from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch
import math
import glob, sys
import pathlib, argparse
from functools import partial
try:
    from training import LitProgressBar
    import training.finetuning as ft
    import architectures as arch
    from architectures.utils import intermediate_layer_names
    from architectures.callbacks import LightningWrapper, LinearEvalWrapper
    from attack.callbacks import AdvCallback
    from data_modules import DATA_MODULES
    import dataset_metadata as dsmd
    from partially_inverted_reps.partial_loss import PartialInversionLoss, PartialInversionRegularizedLoss
    from partially_inverted_reps import DATA_PATH_IMAGENET, DATA_PATH
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--source_dataset', type=str, default=None)
parser.add_argument('--finetuning_dataset', type=str, default='cifar10')
parser.add_argument('--base_dir', type=str, default=None)
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--devices', type=str, default='all')


SEED = 2
NUM_NODES = 1
DEVICES = torch.cuda.device_count()
STRATEGY = DDPPlugin(find_unused_parameters=False) if DEVICES > 1 else None
BASE_DIR = pathlib.Path(__file__).parent.resolve()


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.finetuning_dataset](
        data_dir=DATA_PATH_IMAGENET if 'imagenet' in args.finetuning_dataset else DATA_PATH,
        transform_train=dsmd.TRAIN_TRANSFORMS_TRANSFER_DEFAULT(224),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(224),
        batch_size=args.batch_size, shuffle_train=False)
    dm.init_remaining_attrs(args.source_dataset)

    m1 = arch.create_model(args.model, args.source_dataset, pretrained=True,
                           checkpoint_path=args.checkpoint_path, seed=SEED, 
                           num_classes=dsmd.DATASET_PARAMS[args.source_dataset]['num_classes'],
                           callback=partial(LightningWrapper, 
                                            inference_kwargs={'with_latent': True}, 
                                            dataset_name=args.source_dataset, 
                                            training_params_dataset=args.finetuning_dataset), 
                           loading_function_kwargs={'strict': False} if '_ff' in args.model or '_mrl' in args.model else {})
                           ### keep strict False since some resnets have a strange last layer

    pl_utils.seed.seed_everything(SEED, workers=True)

    filename = f'{BASE_DIR if args.base_dir is None else args.base_dir}/checkpoints/'\
              f'{args.model}-base-{args.source_dataset}-ft-{args.finetuning_dataset}/principal_components'
    filename += f'_{args.append_path}.pt' if args.append_path else '.pt'

    trainer = Trainer(accelerator='gpu', 
                      devices=DEVICES,
                      num_nodes=NUM_NODES,
                      strategy=STRATEGY, 
                      log_every_n_steps=1,
                      auto_select_gpus=True, 
                      deterministic=True,
                      check_val_every_n_epoch=1,
                      num_sanity_val_steps=0,
                      sync_batchnorm=True,
                      callbacks=[LitProgressBar(['loss'])])
    out = trainer.predict(m1, dataloaders=[dm.train_dataloader()])

    if trainer.is_global_zero:
        # perform PCA on full output and save principal components
        latent = out[1]
        print (latent.shape)
        _, _, V = torch.pca_lowrank(latent, 
            q=min(latent.shape[0],latent.shape[1]), 
            center=True)
        torch.save(V, filename)


if __name__=='__main__':
    main()