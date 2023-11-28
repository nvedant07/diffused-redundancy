import torch as ch

from attack.losses import LPNormLossSingleModel
from architectures.inference import inference_with_features

class PartialInversionLoss(LPNormLossSingleModel):

    def __init__(self, lpnorm_type, layer, fraction, mode, seed=None, *args, **kwargs):
        """
        layer (int) : Layer to use for extracting features
        fraction (0.0 < float <= 1.0) : amount of neurons to use in inversions
        mode (str) : 'random' whether we're picking random nodes from the feature vector
                     'opt' optimizing for the 
        """
        super().__init__(lpnorm_type, *args, **kwargs)
        self.layer = layer
        self.fraction = fraction
        self.mode = mode
        self.seed = seed ## needed for mode == 'random'
        self.chosen_neurons = None

    def find_mask(self, rep):
        mask = ch.zeros(rep.shape[-1])
        num_neurons = int(self.fraction * rep.shape[-1])
        if self.mode == 'random':
            ch.manual_seed(self.seed)
            ## masking operation fails on GPU; fix requires making a copy on CPU 
            ## (https://github.com/pytorch/pytorch/issues/61032); better to just do 
            ## masking on CPU and then use the (slower) .to(device) call here
            chosen_neurons = ch.randperm(len(mask))[:num_neurons]
        elif self.mode == 'min':
            chosen_neurons = self.chosen_neurons if self.chosen_neurons is not None else \
                ch.argsort(ch.linalg.norm(rep, axis=0))[:num_neurons]
            if self.chosen_neurons is None:
                self.chosen_neurons = chosen_neurons
        elif self.mode == 'max':
            chosen_neurons = self.chosen_neurons if self.chosen_neurons is not None else \
                ch.argsort(ch.linalg.norm(rep, axis=0))[-num_neurons:]
            if self.chosen_neurons is None:
                self.chosen_neurons = chosen_neurons
        else:
            raise ValueError(f'Mode {self.mode} not valid!')
        mask[chosen_neurons] = 1
        mask = mask.bool().to(rep.device)
        return mask

    def __call__(self, model1, model2, inp, targ1, targ2):
        inp = self._transform_input(inp)
        _, rep1 = inference_with_features(model1, inp, with_latent=True, fake_relu=True)
        mask = self.find_mask(rep1)
        self.unmasked_norm = ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1)
        rep1, targ1 = mask * rep1, mask * targ1
        self.model1_loss_normed = ch.div(ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1), 
                                         ch.norm(targ1, p=self.lpnorm_type, dim=1) + 1e-10)
        self.model1_loss = ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1)
        loss = self.model1_loss_normed

        rep1 = None
        ch.cuda.empty_cache()

        return loss

    def clear_cache(self) -> None:
        self.model1_loss, self.model1_loss_normed, self.unmasked_norm = None, None, None
        self.chosen_neurons = None
        ch.cuda.empty_cache()

    def __repr__(self):
        return f'Model1 Loss: {ch.mean(self.model1_loss)} '\
            f'({ch.mean(self.model1_loss_normed)}), '\
            f'unmasked: {ch.mean(self.unmasked_norm)}'


class PartialInversionRegularizedLoss(PartialInversionLoss):
    """
    In addition to matching the partial neurons this 
    makes sure the other neurons are far away 
    """
    def __init__(self, *args, **kwargs):
        """
        layer (int) : Layer to use for extracting features
        fraction (0.0 < float <= 1.0) : amount of neurons to use in inversions
        mode (str) : 'random' whether we're picking random nodes from the feature vector
                     'opt' optimizing for the 
        """
        super().__init__(*args, **kwargs)
        self.alpha = None # balancing factor

    def __call__(self, model1, model2, inp, targ1, targ2):
        inp = self._transform_input(inp)
        _, rep1 = inference_with_features(model1, inp, with_latent=True, fake_relu=True)
        mask = self.find_mask(rep1)
        self.model1_loss_normed = ch.div(ch.norm(mask * rep1 - mask * targ1, p=self.lpnorm_type, dim=1), 
                                         ch.norm(mask * targ1, p=self.lpnorm_type, dim=1) + 1e-10)
        self.model1_loss = ch.norm(mask * rep1 - mask * targ1, p=self.lpnorm_type, dim=1)
        self.unmasked_norm = ch.div(ch.norm(~mask * rep1 - ~mask * targ1, p=self.lpnorm_type, dim=1), 
                                    ch.norm(~mask * targ1, p=self.lpnorm_type, dim=1) + 1e-10)
        if self.alpha is None:
            self.alpha = ch.mean(self.model1_loss_normed).item()/ch.mean(self.unmasked_norm).item()

        rep1 = None
        ch.cuda.empty_cache()

        loss = self.model1_loss_normed - self.alpha * self.unmasked_norm
        return loss

    def clear_cache(self) -> None:
        self.model1_loss, self.model1_loss_normed = None, None
        self.unmasked_norm, self.regularized_loss_normed = None, None
        ch.cuda.empty_cache()

