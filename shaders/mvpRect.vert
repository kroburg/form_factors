#version 330

layout (location = 0) in vec2 vertex;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;

out highp vec2 vert;

void main() {
    vert = vertex;
    gl_Position = projMatrix * mvMatrix * vec4(vertex, 0.0, 1.0);;
}