import torch
import torch.nn as nn
from math import sqrt


def __getattr(model, name):
    name_list = name.split(".")
    ret = model
    for n in name_list:
        if n.isdigit():
            ret = ret[int(n)]
        else:
            ret = getattr(ret, n)
    return ret


def repair_model_vgg(model, restore):
    W, B = False, False
    if restore == 1:
        W = True
    elif restore == 2:
        B = True
    elif restore == 3:
        W, B = True, True
    elif restore == 12:  # restore to kaiming in
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / (w_shape[1] * w_shape[2] * w_shape[3])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / w_shape[1]
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass
    elif restore == 14:  # restore to kaiming out
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / (w_shape[0] * w_shape[2] * w_shape[3])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / w_shape[0]
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass
    elif restore == 16:  # restore to xavier
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance xaiver
                    w_shape = m.weight.shape
                    aimed_var = 4 / (
                        (w_shape[1] + w_shape[0]) * w_shape[2] * w_shape[3]
                    )
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 4 / (w_shape[1] + w_shape[0])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass

    if W or B:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                lv = repair_conv(m, W)
                if B:
                    name_split = n.split(".")
                    bn_name = (
                        ".".join(name_split[:-1]) + "." + str(int(name_split[-1]) + 1)
                    )
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm2d):
                            # print(f"{bn_name} not correct")
                            continue
                        repair_bn(bn, lv)
                        # print(f"{bn_name} founded")
                    except:
                        # print(f"{bn_name} not found")
                        pass
            elif isinstance(m, nn.Linear):
                lv = repair_fc(m, W)
                if B:
                    name_split = n.split(".")
                    bn_name = (
                        ".".join(name_split[:-1]) + "." + str(int(name_split[-1]) + 1)
                    )
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm1d):
                            # print(f"{bn_name} not correct")
                            continue
                        repair_bn(bn, lv)
                        # print(f"{bn_name} founded")
                    except:
                        # print(f"{bn_name} not found")
                        pass


def repair_model(model, restore):
    W, B = False, False
    if restore == 1:
        W = True
    elif restore == 2:
        B = True
    elif restore == 3:
        W, B = True, True
    elif restore == 12:  # restore to kaiming in
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / (w_shape[1] * w_shape[2] * w_shape[3])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / w_shape[1]
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass
    elif restore == 14:  # restore to kaiming out
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / (w_shape[0] * w_shape[2] * w_shape[3])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 2 / w_shape[0]
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass
    elif restore == 16:  # restore to xavier
        for n, m in model.named_modules():
            try:
                if isinstance(m, nn.Conv2d):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance xaiver
                    w_shape = m.weight.shape
                    aimed_var = 4 / (
                        (w_shape[1] + w_shape[0]) * w_shape[2] * w_shape[3]
                    )
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
                elif isinstance(m, nn.Linear):
                    # new module weight
                    shape = m.weight_orig.data[m.weight_mask == 1].shape
                    m.weight_orig.data[m.weight_mask == 1] = torch.randn(shape)
                    # m.weight_orig.data[m.weight_mask == 0] = 0
                    m.weight.data = m.weight_orig * m.weight_mask
                    # restore variance
                    w_shape = m.weight.shape
                    aimed_var = 4 / (w_shape[1] + w_shape[0])
                    cur_var = m.weight.var().item()
                    m.weight_orig.data *= sqrt(aimed_var / cur_var)
                    m.weight.data = m.weight_orig * m.weight_mask
            except:
                pass

    if W or B:
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                # print(n)
                lv = repair_conv(m, W)
                if B:
                    bn_name = n.replace("conv", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm2d):
                            # print(f"{bn_name} not correct")
                            continue
                        repair_bn(bn, lv)
                        # print(n, lv)
                        # print(f"{bn_name} founded")
                    except:
                        # print(f"{bn_name} not found")
                        pass
            elif isinstance(m, nn.Linear):
                # print(n)
                lv = repair_fc(m, W)
                if B:
                    bn_name = n.replace("fc", "bn")
                    try:
                        bn = __getattr(model, bn_name)
                        if not isinstance(bn, nn.BatchNorm1d):
                            # print(f"{bn_name} not correct")
                            continue
                        repair_bn(bn, lv)
                        # print(f"{bn_name} founded")
                    except:
                        # print(f"{bn_name} not found")
                        pass


def repair_conv(conv, action=True):
    if hasattr(conv, "weight_mask"):
        return __repair_masked_conv(conv, action)
    else:
        return __repair_conv(conv, action)


def __repair_masked_conv(conv, action):
    mean = conv.weight[conv.weight != 0].mean().item()
    var = conv.weight.var().item()
    if action and abs(mean) <= 1e-2:
        conv.weight_orig.data -= mean
        conv.weight.data = conv.weight_orig * conv.weight_mask
    try:
        sn = torch.linalg.norm(conv.weight.view(conv.weight.shape[0], -1), ord=2).item()
    except:
        sn = 1
    # print(sn)
    if action:
        conv.weight_orig.data /= sn
        conv.weight.data = conv.weight_orig * conv.weight_mask
    shape = conv.weight.shape[1] * conv.weight.shape[2] * conv.weight.shape[3]
    tmp = sqrt(0.5 * shape * var)
    if tmp == 0:
        return 1
    lv = sn / tmp
    return lv


def __repair_conv(conv, action):
    mean = conv.weight.mean().item()
    var = conv.weight.var().item()
    if action and abs(mean) <= 1e-2:
        conv.weight.data -= mean
    try:
        sn = torch.linalg.norm(conv.weight.view(conv.weight.shape[0], -1), ord=2).item()
    except:
        sn = 1
    if action:
        conv.weight.data /= sn
    shape = conv.weight.shape[1] * conv.weight.shape[2] * conv.weight.shape[3]
    tmp = sqrt(0.5 * shape * var)
    if tmp == 0:
        return 1
    lv = sn / tmp
    return lv


def repair_fc(fc, action=True):
    if hasattr(fc, "weight_mask"):
        return __repair_masked_fc(fc, action)
    else:
        return __repair_fc(fc, action)


def __repair_masked_fc(fc, action):
    mean = fc.weight[fc.weight != 0].mean().item()
    var = fc.weight.var().item()
    if action and abs(mean) <= 1e-2:
        fc.weight_orig.data -= mean
        fc.weight.data = fc.weight_orig * fc.weight_mask
    try:
        sn = torch.linalg.norm(fc.weight, ord=2).item()
    except:
        sn = 1
    # print(sn)
    if action:
        fc.weight_orig.data /= sn
        fc.weight.data = fc.weight_orig * fc.weight_mask
    tmp = sqrt(0.5 * fc.weight.shape[1] * var)
    if tmp == 0:
        return 1
    lv = sn / tmp
    return lv


def __repair_fc(fc, action):
    mean = fc.weight.mean().item()
    var = fc.weight.var().item()
    if action and abs(mean) <= 1e-2:
        fc.weight.data -= mean
    try:
        sn = torch.linalg.norm(fc.weight, ord=2).item()
    except:
        sn = 1
    if action:
        fc.weight.data /= sn
    tmp = sqrt(0.5 * fc.weight.shape[1] * var)
    if tmp == 0:
        return 1
    lv = sn / tmp
    return lv


def repair_bn(bn, lv):
    bn.weight.data /= lv
    return lv
