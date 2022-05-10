import copy
from functools import partial

import torch
from torch.autograd.functional import jacobian
from torch.nn import Linear


class ModuleParameters:
    def __init__(self, module, grad=False):
        if grad:
            self.parameters = {k: v.grad for k, v in module.named_parameters()}
        else:
            self.parameters = {k: v for k, v in module.named_parameters()}
    
    @staticmethod
    def from_dict(dict):
        params = ModuleParameters(torch.nn.Module()) # Create with empty Module
        params.parameters = {k: v for k, v in dict.items()}
        return params

    def __add__(self, other):
        return ModuleParameters.from_dict(
            {k: v + other.parameters[k] for k, v in self.parameters.items()}
        )

    def __sub__(self, other):
        return ModuleParameters.from_dict(
            {k: v - other.parameters[k] for k, v in self.parameters.items()}
        )
    
    def __mul__(self, other):
        if isinstance(other, ModuleParameters):
            return ModuleParameters.from_dict(
                {k: v * other.parameters[k] for k, v in self.parameters.items()}
            )
        else:
            return other * self

    def __rmul__(self, other):
        return ModuleParameters.from_dict(
            {k: v * other for k, v in self.parameters.items()}
        )

    def __truediv__(self, other):
        if isinstance(other, ModuleParameters):
            return ModuleParameters.from_dict(
                {k: v / other.parameters[k] for k, v in self.parameters.items()}
            )
        else:
            return 1 / other * self

    def __pow__(self, other):
        return ModuleParameters.from_dict(
            {k: v ** other for k, v in self.parameters.items()}
        )
    
    def sum(self):
        s = 0.
        for v in self.parameters.values():
            s += v.sum()
        
        return s
    
    def dot(self, other):
        x = 0.
        for k,v in self.parameters.items():
            x += torch.dot(v.flatten(), other.parameters[k].flatten())
        
        return x


# Utility function to delete a parameter from a model
# based on the name given by the named_parameters function
def delparam(module, name):
    name_list = name.split('.')
    while len(name_list) > 1:
        module = getattr(module, name_list[0])
        name_list = name_list[1:]
    
    delattr(module, name_list[0])


# Utility function to set a parameter to a model
def setparam(module, name, param):
    name_list = name.split('.')
    while len(name_list) > 1:
        module = getattr(module, name_list[0])
        name_list = name_list[1:]
    
    setattr(module, name_list[0], param)


class EWC:
    def __init__(self, module, samples, fn=None, method='Normal'):
        # We need to change the module so make a deepcopy of it
        self.module = copy.deepcopy(module)

        self.mean_params = ModuleParameters(self.module)

        if method == 'Normal':
            self.kernel_diag = self._compute_kernel_diag(samples, fn)
        elif method == 'Batched':
            self.kernel_diag = self._compute_kernel_diag_batched(samples, fn)
        elif method == 'Hooks':
            self.kernel_diag = self._compute_kernel_diag_hooks(samples, fn)
        else:
            raise NotImplementedError(f'Invalid method: {method}')
        
    def _compute_kernel_diag(self, samples, fn):
        self.module.eval()

        n_samples = samples.shape[0]
        
        # Get list of names and parameters
        names, params = list(zip(*self.module.named_parameters()))

        # Pytorch jacobian function copies the tensor objects
        # so we need a function that takes the parameters and
        # uses them so that they are part of the computation graph
        def jac_f(*params):
            # This process doesn't work if the attributes are torch.nn.Parameters
            # so set them as torch.Tensor. This potentially breaks code, although
            # it seems unlikely
            for n, p in zip(names, params):
                delparam(self.module, n)
                setparam(self.module, n, p)

            output = fn(self.module, samples) if fn else self.module(samples)
            return output.squeeze(-1)

        jac = jacobian(jac_f, params)

        kernel_diag = ModuleParameters.from_dict({n: (p**2).mean(dim=0) for n, p in zip(names, jac)})

        return kernel_diag

    def _compute_kernel_diag_batched(self, samples, fn, batch_size=1000):
        self.module.eval()

        n_samples = samples.shape[0]
        
        # Get list of names and parameters
        names, params = list(zip(*self.module.named_parameters()))

        # Pytorch jacobian function copies the tensor objects
        # so we need a function that takes the parameters and
        # uses them so that they are part of the computation graph
        def jac_f(samples, *params):
            # This process doesn't work if the attributes are torch.nn.Parameters
            # so set them as torch.Tensor. This potentially breaks code, although
            # it seems unlikely
            for n, p in zip(names, params):
                delparam(self.module, n)
                setparam(self.module, n, p)

            output = fn(self.module, samples) if fn else self.module(samples)
            return output.squeeze(-1)

        n_batches = (n_samples + batch_size - 1) // batch_size
        kernel_diag = None
        for t in torch.tensor_split(samples, n_batches):
            jac = jacobian(partial(jac_f, t), params)
            if kernel_diag:
                kernel_diag = kernel_diag + ModuleParameters.from_dict({n: (p**2).sum() for n, p in zip(names, jac)})
            else:
                kernel_diag = ModuleParameters.from_dict({n: (p**2).sum() for n, p in zip(names, jac)})

        kernel_diag = kernel_diag / n_samples

        return kernel_diag

    def _compute_kernel_diag_hooks(self, samples, fn):
        self.module.eval()

        handles = []
        for m in self.module.modules():
            if isinstance(m, Linear):
                hf = m.register_forward_hook(self._forward_hook)
                hb = m.register_full_backward_hook(self._backward_hook)
                handles.extend((hf, hb))
        
        self.saved = dict()
        self.grads = dict()

        samples = samples.detach() # Make shallow copy of samples
        samples.requires_grad = True # This is needed in order for the backward 
                                     # computation to propagate beyond the last layer
        out = fn(self.module, samples) if fn else self.module(samples)
        out.sum(dim=0).backward()
        
        kernel_diag = ModuleParameters.from_dict({n: self.grads[p] for n, p in self.module.named_parameters()})
        
        del self.saved
        del self.grads
        for h in handles:
            h.remove()

        return kernel_diag

    def _forward_hook(self, mod, input, output):
        self.saved[mod] = input[0] # input is always a tuple

    def _backward_hook(self, mod, grad_input, grad_output):
        x = self.saved[mod]
        gy = grad_output[0] # grad_output is always a tuple
        self.grads[mod.weight] = torch.mean(torch.bmm(gy.unsqueeze(2), x.unsqueeze(1)) ** 2, dim=0)
        if mod.bias is not None:
            self.grads[mod.bias] = torch.mean(gy**2, dim=0)
    
    def loss(self, other_module):
        other_params = ModuleParameters(other_module)

        return self.kernel_diag.dot((self.mean_params - other_params)**2)
