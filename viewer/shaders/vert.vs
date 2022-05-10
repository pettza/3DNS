#version 330 core
layout (location = 0) in vec2 pos;

uniform ivec2 res;

out vec2 tex_coords;

void main() {
    tex_coords = pos;

    gl_Position = vec4(2*pos - 1, 0., 1.);
}