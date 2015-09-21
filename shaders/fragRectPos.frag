#version 150

in highp vec2 vert;
uniform highp float pos;
out highp vec4 fragColor;

void main() {
    if (vert.x <= pos) {
        fragColor = vec4(vec3(1.0), 0.8);
    } else {
        fragColor = vec4(vec3(1.0), 0.2);
    }
}