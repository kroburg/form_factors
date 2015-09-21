#version 150

in highp vec2 vert;

out highp vec4 fragColor;

void main() {
    vec3 lc = vec3(0, 0, 1);
    vec3 mc = vec3(0, 1, 0);
    vec3 hc = vec3(1, 0, 0);
    float coord = 1 - vert.y;
    if (coord < 0.5) {
        fragColor = vec4(mix(lc, mc, coord * 2), 1.0);
    } else {
        fragColor = vec4(mix(mc, hc, (coord - 0.5) * 2), 1.0);
    }
}