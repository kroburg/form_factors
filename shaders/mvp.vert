#version 330

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec3 normal;
layout (location = 2) in float temp;

// Outputs mv-applied vector
out highp vec3 vert;
// Outputs mv-applied vector
out highp vec3 vertNormal;
// Outputs vertex temperature
out highp float vertTemp;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 cameraMatrix;
uniform mat3 normalMatrix;

void main() {
    vertNormal = normalize(normalMatrix * normal);
    vertTemp = temp;
    highp vec4 curVert = mvMatrix * vec4(vertex, 1.0);
    vert = curVert.xyz;
    gl_Position = projMatrix * cameraMatrix * curVert;
}