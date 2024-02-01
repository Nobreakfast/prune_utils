import torch
import torch.nn as nn


class Unstructured(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, indices):
        self.indices = indices

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if self.indices != None:
            mask.view(-1)[self.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, indices):
        return super(Unstructured, cls).apply(module, name, indices=indices)


## score function
def rand(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.rand_like(module.weight.data).abs().view(-1)
    return score_dict


def randn(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.randn_like(module.weight.data).abs().view(-1)
    return score_dict


def l1_norm(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = module.weight.data.to(torch.device("cpu")).abs().view(-1)
    return score_dict


def snip(model, dataloader):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.zeros_like(module.weight.data).view(-1)
    for i, (input, target) in enumerate(dataloader):
        pass
    return score_dict


def synflow(model, dataloader):
    raise NotImplementedError
