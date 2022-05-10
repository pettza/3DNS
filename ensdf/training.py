import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

import time
import numpy as np
import os
import shutil

from . import loss_functions
from .datasets import RegularizationDataset
from .diff_operators import gradient


def pretrain(model, num_samples, epochs, device):
    model.to(device)
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    reg_dataset = RegularizationDataset(num_samples, device)
    
    for epoch in range(epochs):    
        samples = reg_dataset.sample()
        points = samples['points']
        reg_sdf = model(points)
        norms = torch.norm(points, dim=1, keepdim=True)
        loss = torch.mean(torch.abs(norms - reg_sdf))
        
        optim.zero_grad()
        loss.backward()
        optim.step()


def train_sdf(model, surface_dataset, epochs, lr, epochs_til_checkpoint,
              model_dir, device, pretrain_epochs=0, regularization_samples=0,
              include_empty_space_loss=True, ewc=None):
    model.to(device)
    model.train()
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    reg_dataset = RegularizationDataset(regularization_samples, device) if regularization_samples > 0 else None

    if os.path.exists(model_dir):
        prompt = f'The model directory {model_dir} exists. Overwrite? (y/n): '
        
        val = input(prompt).lower()
        while val not in {'y', 'n'}:
            val = input(prompt).lower()
        
        if val == 'y':
            shutil.rmtree(model_dir)
        else:
            print('Cannot proceed without valid directory')
            exit()

    os.makedirs(model_dir, exist_ok=True)

    summary_dir = os.path.join(model_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    writer = SummaryWriter(summary_dir)

    pretrain(model, regularization_samples, pretrain_epochs, device)
    
    total_steps = 0
    with tqdm(total=epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            start_time = time.time()

            samples = surface_dataset.sample()
            sample_points = samples['points']
            sample_points.requires_grad = True
            gt_normals = samples['normals']
            gt_sdf = samples['sdf']
            
            sample_sdf = model(sample_points)
            sample_normals = gradient(sample_sdf, sample_points)
            
            losses = {
                'sdf_contraint': loss_functions.sdf_loss(sample_sdf, gt_sdf) * 1.5e3,
                'normal_constraint': loss_functions.normal_loss(sample_normals, gt_normals) * 5e1,
                'implicit_reg_constraint': loss_functions.implicit_reg_loss(sample_normals) * 2.5e1
            }

            if reg_dataset:
                reg_samples = reg_dataset.sample()
                reg_points = reg_samples['points']
                reg_points.requires_grad = True

                reg_sdf = model(reg_points)
                reg_normals = gradient(reg_sdf, reg_points) 

                losses['implicit_reg_constraint'] += loss_functions.implicit_reg_loss(reg_normals) * 2.5e1

                if include_empty_space_loss:
                    losses['inter_constraint'] = loss_functions.empty_space_loss(reg_sdf) * 5e1

            total_loss = 0.
            for loss_name, loss in losses.items():
                writer.add_scalar(loss_name, loss, epoch)
                total_loss += loss

            train_losses.append(total_loss.item())
            writer.add_scalar('total_train_loss', total_loss, epoch)
            
            if not epoch % epochs_til_checkpoint and epoch:
                iteration_time = time.time() - start_time
                tqdm.write(f'Epoch {epoch}, Total loss {total_loss:0.6f}, iteration time {iteration_time:0.6f}')
                
                model.save(os.path.join(checkpoints_dir, f'model_epoch_{epoch}.pth'))
                np.savetxt(os.path.join(checkpoints_dir, f'train_losses_epoch_{epoch}.txt'),
                           np.array(train_losses)
                )
            
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            pbar.update(1)

        model.save(os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
