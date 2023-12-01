## Code for Diffused Redundancy [NeurIPS2023](https://arxiv.org/abs/2306.00183)

Code to reproduce results from our NeurIPS 2023 paper: "Diffused Redundancy in Pre-trained Representations".

Relies on many wrapper written to extended (PyTorch)Lightning, that is included here as a submodule: [deep-learning-base](https://github.com/nvedant07/deep-learning-base/).

Install python >= 3.7.9 all requirements in requirements.txt.

### Finetuning models scripts

We provide easy-to-use scripts to run finetuning on different (randomly) picked neurons from a representation.
These can be found under ``scripts/finetuning_*.sh``

For finetuning on middle layers see ``scripts/layerwise_finetuning_*.sh``

### Comparison with PCA and random projections

Before being able to run PCA comparisons, you need to compute the principal components on the training set of the downstream task. This can be done using ``calculate_pca.py``. Some examples of how to use this script can be found in ``scripts/pca_*.sh``.

To use these computed principal components to then train a linear probe for downstream tasks, see ``scripts/finetuning_*-pca.sh`` (top principal components) and ``scripts/finetuning_*-pca-least.sh`` (bottom principal components).

For linear probes trained on random projection of the representation (into a lower dimensional space) see ``scripts/finetuning_*-randproj.sh``.

### Model weights

We've attached pre-trained model weights (see release files) to aid reproducibility.

## Citation

If you find our work useful please cite our paper:

```
@inproceedings{nanda2023diffused,
    title={Diffused Redundancy in Pre-trained Representations},
    author={Nanda, Vedant and Speicher, Till and Dickerson, John P. and Feizi, Soheil and Gummadi, Krishna P. and Weller, Adrian},
    booktitle={NeurIPS},
    year={2023}
}
```