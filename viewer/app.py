import sys
import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pyglet
from pyglet.window import key, mouse
from OpenGL.GL import *

from viewer_utils import ScreenTexture, BrushRenderer

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
import ensdf.modules as modules
from ensdf.brushes import SimpleBrush
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch
from ensdf.rendering.camera import OrbitingCamera
from ensdf.rendering.shading import phong_shading, shade_normals
from ensdf.datasets import SDFEditingDataset
from ensdf.training import train_sdf
from ensdf.utils import get_cuda_if_available


class ENSDFWindow(pyglet.window.Window):
    def __init__(self, resolution=(1280, 720), epochs_per_edit=80):
        super().__init__(caption="SDF Viewer", width=resolution[0], height=resolution[1])

        self.resolution = resolution
        self.epochs_per_edit = epochs_per_edit

        script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        shader_dir = os.path.join(script_dir, 'shaders')

        self.screen_tex = ScreenTexture(
            resolution,
            os.path.join(shader_dir, 'screen_texture_vert.vs'),
            os.path.join(shader_dir, 'screen_texture_frag.fs')
        )
        
        self.device = get_cuda_if_available()

        self.model = modules.Siren.load(sys.argv[1])
        self.model.to(self.device)
        self.prev_model = self.model

        self.aabb = AABB([0., 0., 0.], [1., 1., 1.], device=self.device)
        self.camera = OrbitingCamera(np.deg2rad(60), self.resolution, np.deg2rad(0), np.deg2rad(90), 3.2)

        self.background_color = torch.tensor([0.2, 0.2, 0.2], device=self.device).view(1, 1, 3)
        self.light_colors = torch.tensor([1., 1., 1.], device=self.device)

        self.brush_types = ['linear', 'cubic', 'quintic', 'exp']
        self.brush_type_idx = 2 # quintic

        self.brush = SimpleBrush(
            brush_type=self.brush_types[self.brush_type_idx],
            radius=0.08,
            intensity=0.04
        )
        
        self.edit_dataset = SDFEditingDataset(
            self.model, self.device, self.brush,
            num_model_samples=120_000
        )
        self.lr = 1e-4
        
        self.brush_renderer = BrushRenderer(
            self.camera, self.brush,
            os.path.join(shader_dir, 'brush_vert.vs'),
            os.path.join(shader_dir, 'brush_frag.fs')
        )
        
        self.mouse_pos = (0, 0)

        self.retrace = True

    def print_brush(self):
        print(f'Radius: {self.brush.radius:0.2f} | ', sep='')
        print(f'Intensity: {self.brush.intensity:0.2f} | ', sep='')
        print(f'Type: {self.brush.brush_type}')

    def on_draw(self):
        self.clear()

        if self.retrace:
            origins, directions = self.camera.generate_rays()
            origins = origins.to(self.device)
            directions = directions.to(self.device)

            self.traced_positions, self.traced_normals, self.traced_sdf, self.traced_hit = raymarch(
                self.model, self.aabb, origins, directions, num_iter=80
            )
            F.normalize(self.traced_normals, dim=-1, out=self.traced_normals)
            
            eye_pos = self.camera.position.to(self.device)
            light_pos = eye_pos # eye coming from camera position

            colors = phong_shading(
                self.traced_positions, self.traced_normals, light_pos, self.light_colors, eye_pos
            )
            # colors = shade_normals(self.traced_normals)

            image = torch.where(self.traced_hit, colors, self.background_color).transpose_(0, 1)
            torch.clamp(image, min=0., max=1., out=image)

            tex_data = image.detach().cpu().numpy()

            self.screen_tex.update_texture(tex_data)
            
            self.retrace = False
        
        self.screen_tex.render()

        x, y = self.mouse_pos
        if self.valid_interaction(x, y):
            self.brush.set_interaction(
                self.traced_positions[x, y:y+1],
                self.traced_normals[x, y:y+1],
                self.traced_sdf[x, y:y+1]
            )
            self.brush_renderer.render()

    def valid_interaction(self, mouse_x, mouse_y):
        cond = (
            0 <= mouse_x < self.resolution[0] and
            0 <= mouse_y < self.resolution[1] and
            self.traced_hit[mouse_x, mouse_y]
        )
        return cond

    def on_key_press(self, symbol, modifiers):
        if modifiers & key.MOD_ALT:
            if symbol == key.UP:
                self.brush_type_idx += 1
                self.brush_type_idx %= len(self.brush_types)
                self.brush.brush_type = self.brush_types[self.brush_type_idx]
                self.print_brush()
            elif symbol == key.DOWN:
                self.brush_type_idx -= 1
                self.brush_type_idx %= len(self.brush_types)
                self.brush.brush_type = self.brush_types[self.brush_type_idx]
                self.print_brush()
        elif modifiers & key.MOD_CTRL:
            if symbol == key.UP:
                self.brush.radius += 0.01
                self.print_brush()
            elif symbol == key.DOWN:
                self.brush.radius -= 0.01
                self.print_brush()
            elif symbol == key.LEFT:
                self.brush.intensity -= 0.01
                self.print_brush()
            elif symbol == key.RIGHT:
                self.brush.intensity += 0.01
                self.print_brush()
            elif symbol == key.ENTER:
                model_path = input('Model path: ').strip()
                self.model.save(model_path)
            elif symbol == key.Z:
                if self.model is not self.prev_model:
                    self.model = self.prev_model
                    self.edit_dataset.update_model(self.model, sampler_iters=10)
                    self.retrace = True
        else:    
            if symbol == key.UP:
                self.camera.theta -= np.pi / 10
                self.retrace = True
            elif symbol == key.DOWN:
                self.camera.theta += np.pi / 10
                self.retrace = True
            elif symbol == key.LEFT:
                self.camera.phi -= np.pi / 10
                self.retrace = True
            elif symbol == key.RIGHT:
                self.camera.phi += np.pi / 10
                self.retrace = True
            elif symbol == key.ENTER:
                img = np.flipud(self.screen_tex.tex_data)
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                filename = input('Image filename: ').strip()
                pil_img.save(filename)
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            self.camera.radius *= 0.9
            self.retrace = True
        elif scroll_y < 0:
            self.camera.radius *= 1/0.9
            self.retrace = True
    
    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)
    
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT and self.valid_interaction(x, y):
            self.prev_model = deepcopy(self.model)
            train_sdf(
                model=self.model,
                surface_dataset=self.edit_dataset,
                lr=self.lr,
                epochs=self.epochs_per_edit,
                device=self.device,
                regularization_samples=120_000
            )
            self.edit_dataset.update_model(self.model)
            self.retrace = True


def main():    
    window = ENSDFWindow()
    pyglet.app.run()


if __name__ == '__main__':
    main()
