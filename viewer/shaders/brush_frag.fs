#version 330 core
out vec4 FragColor;

in vec2 plane_pos;

uniform float radius;

void main() {
    float thickness = 0.05 * radius;

    float l = length(plane_pos);
    float d = abs(l - radius);
    
    float edge = smoothstep(thickness, 0., d);
    float disk = smoothstep(radius, 0., l);
    float center = smoothstep(radius * 0.05, 0., l);

    vec4 edge_col = vec4(0.05, 0.1, 0.8, 1.0) * edge;
    vec4 disk_col = vec4(0.7, 0.7, 0.7, 0.5) * disk;
    vec4 center_col = vec4(0., 0., 0., 1.0) * center;
    vec4 col = mix(edge_col + disk_col, center_col, center_col.w);
    
    FragColor = col;
}
