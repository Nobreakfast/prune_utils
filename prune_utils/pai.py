import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class UnstructuredIndice(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, indices):
        self.indices = indices

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if self.indices != None:
            mask[self.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, indices):
        return super(UnstructuredIndice, cls).apply(module, name, indices=indices)


## score function
def rand(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.rand_like(module.weight.data).abs()
    return score_dict


def randn(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.randn_like(module.weight.data).abs()
    return score_dict


def l1_norm(model):
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = module.weight.data.to(torch.device("cpu")).abs()
    return score_dict


def snip(model, dataloader):
    device = next(model.parameters()).device
    score_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.zeros_like(module.weight.data)
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        torch.nn.functional.cross_entropy(output, target).backward()

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                score_dict[name] += (
                    module.weight.grad.data.detach().abs()
                    * module.weight.data.detach().abs()
                )
        model.zero_grad()
        break
    return score_dict


def synflow(model, example_data):
    @torch.no_grad()
    def linearize(model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = module.weight.data.abs()

    @torch.no_grad()
    def nonlinearize(model, signs_dict):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = signs_dict[name] * module.weight.data

    device = next(model.parameters()).device
    score_dict = {}
    sign_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] = torch.zeros_like(module.weight.data)
            sign_dict[name] = torch.sign(module.weight.data).detach()

    input_dim = list(example_data[0, :].shape)
    inputs = torch.ones([1] + input_dim).to(device)

    linearize(model)
    model.eval()
    output = model(inputs)
    torch.sum(output).backward()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            score_dict[name] += (
                module.weight.grad.data.detach().abs()
                * module.weight.data.detach().abs()
            )
    model.zero_grad()
    model.train()
    nonlinearize(model, sign_dict)
    return score_dict


def apply_prune(model, score_dict, threshold):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            indices = score_dict[name] < threshold
            UnstructuredIndice.apply(module, "weight", indices)


def cal_threshold(score_dict, ratio):
    all_scores = torch.cat([torch.flatten(x) for x in score_dict.values()])
    num_params_to_keep = int(len(all_scores) * (1 - ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    return threshold[-1]


def cal_sparsity(model):
    num_zeros = 0
    num_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            num_zeros += module.weight.data.numel() - module.weight.data.nonzero().size(
                0
            )
            num_elements += module.weight.data.numel()
    return num_zeros / num_elements


def remove_mask(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.remove(module, "weight")


if __name__ == "__main__":

    class fake_model(nn.Module):
        def __init__(self):
            super(fake_model, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 12, 5)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            return x

    model = fake_model()
    print(cal_sparsity(model))
    score_dict = rand(model)
    threshold = cal_threshold(score_dict, 0.5)
    apply_prune(model, score_dict, threshold)
    print(cal_sparsity(model))

    # fake datasets with input and fake labels
    train_loader = torch.utils.data.DataLoader(
        torch.randn(100, 3, 28, 28), batch_size=10
    )

    # model = fake_model()
    # score_dict = snip(model, train_loader)
    # threshold = cal_threshold(score_dict, 0.5)
    # apply_prune(model, score_dict, threshold)
    # print(cal_sparsity(model))

    model = fake_model()
    score_dict = synflow(model, train_loader)
    threshold = cal_threshold(score_dict, 0.5)
    apply_prune(model, score_dict, threshold)
    print(cal_sparsity(model))