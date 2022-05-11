import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pyglet
from pyglet.window import key
from OpenGL.GL import *

from screen_texture import ScreenTexture

sys.path.append( os.path.dirname( os.path.dirname(os.path.abspath(__file__) ) ) )
import ensdf.modules as modules
from ensdf.aabb import AABB
from ensdf.raymarching import raymarch
from ensdf.rendering.camera import OrbitingCamera
from ensdf.rendering.shading import phong_shading, shade_normals
from ensdf.utils import get_cuda_if_available


def main():
    resolution = (1280, 720)
    window = pyglet.window.Window(
        caption="SDF Viewer",
        width=resolution[0], height=resolution[1]
    )

    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    shader_dir = os.path.join(script_dir, "shaders")

    screen_tex = ScreenTexture(resolution, shader_dir)
    
    device = get_cuda_if_available()

    model = modules.Siren.load(sys.argv[1])
    model.to(device)
    
    aabb = AABB([0., 0., 0.], [1., 1., 1.], device=device)
    camera = OrbitingCamera(np.deg2rad(60), (1280, 720), np.deg2rad(0), np.deg2rad(80), 3.2)

    background_color = torch.tensor([0.2, 0.2, 0.2], device=device).view(1, 1, 3)
    light_colors = torch.tensor([1., 1., 1.], device=device)

    @window.event
    def on_draw():
        window.clear()

        origins, directions = camera.generate_rays()
        origins = origins.to(device)
        directions = directions.to(device)

        positions, normals, sdf, hit = raymarch(model, aabb, origins, directions, num_iter=80)
        F.normalize(normals, dim=-1, out=normals)
        
        light_pos = camera.position.to(device)

        colors = phong_shading(positions, normals, light_pos, light_colors, camera.position.to(device))

        image = torch.where(hit, colors, background_color).transpose_(0, 1)
        torch.clamp(image, min=0., max=1., out=image)

        tex_data = image.detach().cpu().numpy()

        screen_tex.update_texture(tex_data)
        screen_tex.render()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.UP:
            camera.theta -= np.pi / 10
        elif symbol == key.DOWN:
            camera.theta += np.pi / 10
        elif symbol == key.LEFT:
            camera.phi -= np.pi / 10
        elif symbol == key.RIGHT:
            camera.phi += np.pi / 10
        elif symbol == key.ENTER:
            img = np.flipud(screen_tex.tex_data)
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            filename = input('Image filename: ').strip()
            pil_img.save(filename)
    
    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        if scroll_y > 0:
            camera.radius *= 0.9
        elif scroll_y < 0:
            camera.radius *= 1/0.9

    pyglet.app.run()

    screen_tex.cleanup()


if __name__ == '__main__':
    main()
