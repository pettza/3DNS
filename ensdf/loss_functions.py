import torch
import torch.nn.functional as F


def sdf_loss(pred_sdf, gt_sdf):
    return torch.mean(torch.abs(pred_sdf - gt_sdf))


def normal_loss(pred_normals, gt_normals):
    return torch.mean(1 - F.cosine_similarity(pred_normals, gt_normals, dim=-1))


def empty_space_loss(sdf):
    return torch.mean(torch.exp(-1e2 * torch.abs(sdf)))


def implicit_reg_loss(normals):
    return torch.mean(torch.abs(normals.norm(dim=-1) - 1))
