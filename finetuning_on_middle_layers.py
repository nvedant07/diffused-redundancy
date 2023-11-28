### as opposed to finetuning.py, here the classification head is slapped 
### onto intermediate layer representations as opposed to just the final layer

from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
import torch
import torch.nn as nn
from timm.models.fx_features import GraphExtractNet
import math, glob
import copy
import pathlib, argparse
from functools import partial
from typing import List, OrderedDict, Union
try:
    from training import LitProgressBar, NicerModelCheckpointing
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
parser.add_argument('--finetune_mode', type=str, default='linear')
parser.add_argument('--base_dir', type=str, default=None)
parser.add_argument('--save_every', type=int, default=20)
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--step_lr', type=float, default=None)
parser.add_argument('--warmup_steps', type=int, default=None)
parser.add_argument('--gradient_clipping', type=float, default=0.)
parser.add_argument('--layers', nargs='+', type=int, default=None)
### specific to partial representation
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--fraction', type=float, default=None)
parser.add_argument('--seed', type=int, default=2)

SEED = 2
NUM_NODES = 1
DEVICES = torch.cuda.device_count()
STRATEGY = DDPPlugin(find_unused_parameters=False) if DEVICES > 1 else None
BASE_DIR = pathlib.Path(__file__).parent.resolve()


def lightningmodule_callback(args):
    if args.finetune_mode == 'linear':
        return LinearEvalWrapper
    elif args.finetune_mode == 'full':
        return ft.CosineLRWrapper
    else:
        return LightningWrapper


def main(args=None):
    if args is None:
        args = parser.parse_args()

    dm = DATA_MODULES[args.finetuning_dataset](
        data_dir=DATA_PATH_IMAGENET if 'imagenet' in args.finetuning_dataset else DATA_PATH,
        transform_train=dsmd.TRAIN_TRANSFORMS_TRANSFER_DEFAULT(224),
        transform_test=dsmd.TEST_TRANSFORMS_DEFAULT(224),
        batch_size=args.batch_size)
    dm.init_remaining_attrs(args.source_dataset)

    total_steps = math.ceil(len(dm.train_dataloader())/DEVICES) * args.epochs
    print (f'Total Steps: {total_steps}')
    if args.finetune_mode == 'full' and args.warmup_steps is None:
        args.__setattr__('warmup_steps', int(0.05 * total_steps))
    m1 = arch.create_model(args.model, args.source_dataset, pretrained=True,
                           checkpoint_path=args.checkpoint_path, seed=SEED, 
                           num_classes=dsmd.DATASET_PARAMS[args.source_dataset]['num_classes'],
                           callback=partial(lightningmodule_callback(args),
                                            dataset_name=args.source_dataset,
                                            optimizer=args.optimizer,
                                            step_lr=args.step_lr,
                                            lr=args.lr,
                                            warmup_steps=args.warmup_steps,
                                            total_steps=total_steps)) ## assign mean and std from source dataset
    original_model = copy.deepcopy(m1.model)
    layer_names = intermediate_layer_names(m1.model)
    if args.layers is None:
        args.__setattr__('layers', list(range(len(layer_names)))[:-1]) ## all but the penultimate layer

    for layer_idx in args.layers:
        
        dirpath = f'{BASE_DIR if args.base_dir is None else args.base_dir}/checkpoints/'\
                f'{args.model}-base-{args.source_dataset}-ft-{args.finetuning_dataset}/'
        if args.mode is not None:
            dirpath += f'frac-{args.fraction:.5f}-mode-{args.mode}-seed-{args.seed}-'
        dirpath += f'ftmode-{args.finetune_mode}-lr-{m1.lr}-bs-{dm.batch_size}-{args.append_path}'
        
        trained_model = glob.glob(f'{dirpath}/*-{layer_names[layer_idx]}-topk=1.ckpt')
        if len(trained_model) > 0 and int(trained_model[0].split('epoch=')[1].split('-')[0]) >= 25:
            print (f'A trained model already exists for {layer_names[layer_idx]}, {trained_model[0].split("/")[-1]}')
            continue

        print (f'Working on layer: {layer_idx}')
        chopped_model = GraphExtractNet(copy.deepcopy(original_model), layer_names[layer_idx:layer_idx+1])
        m1.model = nn.Sequential(
            OrderedDict([
                ('feature_extractor', chopped_model), 
                ('flatten', nn.Flatten(1))
                ]))
        new_layer = ft.setup_model_for_finetuning(
            m1.model,
            dsmd.DATASET_PARAMS[args.finetuning_dataset]['num_classes'],
            args.mode, args.fraction, args.seed, 
            inplace=True, infer_features=True)
        if hasattr(new_layer, 'neuron_indices'):
            m1.__setattr__('on_save_checkpoint', 
                lambda checkpoint: checkpoint.update([['neuron_indices', new_layer.neuron_indices]]))

        pl_utils.seed.seed_everything(args.seed, workers=True)

        if args.finetune_mode == 'full':
            dirpath += f'-warmup-{args.warmup_steps}'
        checkpointer = NicerModelCheckpointing(
            dirpath=dirpath, 
            filename='{epoch}' + f'-{layer_names[layer_idx]}', 
            every_n_epochs=args.save_every, 
            save_top_k=1, 
            save_last=False,
            verbose=True,
            mode='max', 
            monitor='val_acc1',
            save_partial=ft.get_param_names(m1.model, args.finetune_mode))

        trainer = Trainer(accelerator='gpu', 
                        devices=DEVICES,
                        num_nodes=NUM_NODES,
                        strategy=STRATEGY, 
                        log_every_n_steps=1,
                        auto_select_gpus=True, 
                        deterministic=True,
                        max_epochs=args.epochs,
                        check_val_every_n_epoch=1,
                        num_sanity_val_steps=0,
                        sync_batchnorm=True,
                        gradient_clip_val=args.gradient_clipping,
                        callbacks=[
                            LitProgressBar(['loss', 
                                            'running_train_acc', 
                                            'running_val_acc']), 
                            checkpointer, 
                            ft.FinetuningCallback(args.finetune_mode)])
        trainer.fit(m1, datamodule=dm)


if __name__=='__main__':
    main()