import torch
import torch.nn.functional as F

from ..diff_operators import gradient
from ..geoutils import project_on_surface

from .primitives import sample_uniform_disk


SAMPLES_PER_ITER = 40000
INIT_REFINEMENT_STEPS = 7       # Refinement steps when initializing
SAMPLING_REFINEMENT_STEPS = 7   # Refinement steps when sampling
REFINEMENT_THRESHOLD = 0.03
ACCEPTANCE_THRESHOLD = 0.009
SIGMA_PERTURB = 0.04


class SDFSampler:
    def __init__(self, model, device, num_samples, burnout_iters=10):
        self.model = model
        self.device = device
        self.num_samples = num_samples
        self.first_yield = True  # Used in __next__

        self.samples        = torch.zeros(0, 3, device=self.device)
        self.sample_sdf     = torch.zeros(0, 1, device=self.device)
        self.sample_normals = torch.zeros(0, 3, device=self.device)

        self.extend_samples()

        self.burnout(burnout_iters)

    def burnout(self, n_iters):
        for i in range(n_iters):
            next(self)

    def extend_samples(self):
        if self.samples.shape[0] == 0:
            iter_samples = torch.rand(
                SAMPLES_PER_ITER, 3,
                dtype=torch.float,
                device=self.device,
                requires_grad=True
            )
            iter_samples = iter_samples * 2 - 1
        else:
            indices = torch.randint(self.samples.shape[0], (SAMPLES_PER_ITER,), device=self.device)
            iter_samples = torch.index_select(self.samples, dim=0, index=indices)
            # Perturb samples for next iteration
            iter_samples.add_(SIGMA_PERTURB * torch.randn(iter_samples.shape, device=self.device))

        while self.samples.shape[0] < self.num_samples:
            iter_samples, sdf_pred, iter_samples_grad = project_on_surface(
                self.model,
                iter_samples,
                num_steps=INIT_REFINEMENT_STEPS
            )
            udf_pred = torch.abs(sdf_pred).squeeze()
            
            inside_BB = torch.max(torch.abs(iter_samples), dim=1)[0] < 1.

            accepted_samples = iter_samples[(udf_pred < ACCEPTANCE_THRESHOLD) & inside_BB]
            self.samples = torch.vstack((self.samples, accepted_samples))
            
            # Keep samples that are close enough
            iter_samples = iter_samples[(udf_pred < REFINEMENT_THRESHOLD) & inside_BB]
            # From those choose randomly with replacements those to perturb
            indices = torch.randint(iter_samples.shape[0], (SAMPLES_PER_ITER,), device=self.device)
            iter_samples = torch.index_select(iter_samples, dim=0, index=indices)
            # Perturb samples for next iteration
            iter_samples.add_(SIGMA_PERTURB * torch.randn(iter_samples.shape, device=self.device))
        
        # Keep only NUM_SAMPLES samples
        shuffled_idx = torch.randperm(self.samples.shape[0])
        self.samples = self.samples[shuffled_idx[:self.num_samples]]

        self.compute_sdf_normals()

    def compute_sdf_normals(self):
        self.samples.requires_grad = True

        self.sample_sdf = self.model(self.samples)

        self.sample_normals = gradient(self.sample_sdf, self.samples)

        self.samples        = self.samples.detach()
        self.sample_sdf     = self.sample_sdf.detach()
        self.sample_normals = self.sample_normals.detach()

    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.first_yield:
            self.samples.add_(SIGMA_PERTURB * sample_uniform_disk(self.sample_normals, 1).squeeze())
            self.samples, self.sample_sdf, self.sample_normals = project_on_surface(
                self.model,
                self.samples, 
                num_steps=SAMPLING_REFINEMENT_STEPS
            )
            udf_pred = torch.abs(self.sample_sdf.squeeze())
            
            inside_BB = torch.max(torch.abs(self.samples), dim=1)[0] < 1.
            cond = (udf_pred < ACCEPTANCE_THRESHOLD) & inside_BB
            self.samples = self.samples[cond]
            self.sample_sdf = self.sample_sdf[cond]
            self.sample_normals = self.sample_normals[cond]
            if self.samples.shape[0] < self.num_samples:
                self.extend_samples()
        
        self.first_yield = False
        return {
            'points': self.samples.clone(),
            'normals': self.sample_normals.clone(),
            'sdf': self.sample_sdf.clone()
        }
