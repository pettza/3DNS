#version 330 core

out vec4 FragColor;

uniform sampler2D tex;

in vec2 tex_coords;

void main() {
    FragColor = texture(tex, tex_coords);
}
