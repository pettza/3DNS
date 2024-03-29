from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    """
    Linear layer with sine non-linearity
    omega_0 is the factor explained in the SIREN paper
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        weight_norm=False,
        is_first=False,
        omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
    
    def add_weight_norm(self):
        if not self.weight_norm:
            self.linear = torch.nn.utils.weight_norm(self.linear, name='weight')
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            self.linear = torch.nn.utils.remove_weight_norm(self.linear, name='weight')
            self.weight_norm = False

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        hidden_layers=None,
        outermost_linear=False,
        weight_norm=False,
        first_omega_0=30,
        hidden_omega_0=30
    ):
        super().__init__()
        
        # If hidden_features is not a list, make it a list whose elements are all
        # equal to hidden_features and of length hidden_layers, so that it can be
        # handled in the same way
        if not isinstance(hidden_features, list):
            if hidden_layers is not None:
                hidden_features = [hidden_features] * hidden_layers
            else:
                raise ValueError("If hidden_features is not a list, hidden_layers should be specified.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.outermost_linear = outermost_linear
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        
        self.net = []
        self.features = [in_features] + hidden_features + [out_features]

        self.net.append(
            SineLayer(
                self.features[0], self.features[1],
                is_first=True,
                omega_0=first_omega_0
            )
        )
        for f_in, f_out in zip(self.features[1:-2], self.features[2:-1]):
            self.net.append(
                SineLayer(
                    f_in, f_out,
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
        
        if outermost_linear:
            final_linear = nn.Linear(self.features[-2], self.features[-1])
            
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / self.features[-2]) / hidden_omega_0, 
                    np.sqrt(6 / self.features[-2]) / hidden_omega_0
                )
                
            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    self.features[-2], self.features[-1],
                    is_first=False,
                    omega_0=hidden_omega_0
                )
            )
          
        self.net = nn.Sequential(*self.net)

        self.weight_norm = False
        if weight_norm:
            self.add_weight_norm()

    def add_weight_norm(self):
        if not self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.add_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.weight_norm(mod, name='weight')
            
            self.weight_norm = True
    
    def remove_weight_norm(self):
        if self.weight_norm:
            for i, mod in enumerate(self.net):
                if isinstance(mod, SineLayer):
                    mod.remove_weight_norm()
                else:
                    self.net[i] = torch.nn.utils.remove_weight_norm(mod, name='weight')
            
            self.weight_norm = False

    def init_parameters(self):
        weight_norm = self.weight_norm
        if weight_norm:
            self.remove_weight_norm()
        
        self.net[0].linear.reset_parameters()
        self.net[0].init_weights()
        for mod in self.net[1:]:
            if isinstance(mod, SineLayer):
                mod.linear.reset_parameters()
                mod.init_weights()
            else:
                mod.reset_parameters()
                with torch.no_grad():
                    mod.weight.uniform_(
                        -np.sqrt(6 / self.features[-2]) / self.hidden_omega_0, 
                        np.sqrt(6 / self.features[-2]) / self.hidden_omega_0
                    )

        if weight_norm:
            self.add_weight_norm()

    def freeze_parameters(self):
        for param in self.parameters():
            param.required_grad = False

    def ufreeze_parameters(self):
        for param in self.parameters():
            param.required_grad = True

    def forward(self, model_input):
        return self.net(model_input)

    def forward_with_activations(self, model_input, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = model_input['coords']
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return {'model_in': x, 'model_out': activations.popitem(), 'activations': activations}
    
    def __deepcopy__(self, memo=None):
        weight_norm = self.weight_norm
        device = list(self.parameters())[0].device
        self.remove_weight_norm()
        
        c = self.__class__(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            outermost_linear=self.outermost_linear,
            weight_norm=self.weight_norm,
            first_omega_0=self.first_omega_0,
            hidden_omega_0=self.hidden_omega_0
        )
        c.load_state_dict(deepcopy(self.state_dict(), memo))
        c.to(device)

        if weight_norm:
            self.add_weight_norm()
            c.add_weight_norm()
        
        return c

    def get_num_bytes(self):
        size_of_float = 4

        size = 0
        for f_in, f_out in zip(self.features[:-1], self.features[1:]):
            size += f_in * f_out + f_out
        
        size *= size_of_float

        return size

    def save(self, path):
        torch.save(
            {
                'in_features':      self.in_features,
                'out_features':     self.out_features,
                'hidden_features':  self.hidden_features,
                'outermost_linear': self.outermost_linear,
                'first_omega_0':    self.first_omega_0,
                'hidden_omega_0':   self.hidden_omega_0,
                'weight_norm':      self.weight_norm,
                'state_dict':       self.state_dict()
            }, path
        )

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        model = Siren(
            in_features=checkpoint['in_features'],
            hidden_features=checkpoint['hidden_features'],
            out_features=checkpoint['out_features'],
            outermost_linear=checkpoint['outermost_linear'],
            first_omega_0=checkpoint['first_omega_0'],
            hidden_omega_0=checkpoint['hidden_omega_0'],
            weight_norm=checkpoint['weight_norm']
        )
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
