#version 150

in vec3 vertex;
in vec3 normal;

// Outputs mv-applied vector
out highp vec3 vert;
// Outputs mv-applied vector
out highp vec3 vertNormal;

uniform mat4 projMatrix;
uniform mat4 mvMatrix;
uniform mat4 cameraMatrix;
uniform mat3 normalMatrix;

void main() {
    vertNormal = normalize(normalMatrix * normal);
    highp vec4 curVert = mvMatrix * vec4(vertex, 1.0);
    vert = curVert.xyz;
    gl_Position = projMatrix * cameraMatrix * curVert;
}