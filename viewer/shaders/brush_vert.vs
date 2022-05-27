#version 330 core
layout (location = 0) in vec2 pos;

uniform float radius;
uniform vec2 f;
uniform vec3 inter;
uniform vec3 cam_pos;
uniform mat3 view;
uniform vec3 t1, t2;

out vec2 plane_pos;

void main() {
    plane_pos = pos * (1.1 * radius);
    vec3 t = plane_pos.x * t1 + plane_pos.y * t2 + inter - cam_pos;
    vec4 p = vec4(view * t, 1.0);
    p.w = p.z;
	p.xy *= f.xy;
    p.x = -p.x;
	p.w = p.z;
    gl_Position = p;
}
