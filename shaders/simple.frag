#version 150

in highp vec3 vert;
in highp vec3 vertNormal;

uniform highp vec3 lightPos;
uniform highp vec3 frontColor;
uniform highp vec3 backColor;
uniform highp vec3 cameraPos;

out highp vec4 fragColor;

void main() {
    // This is simple diffuse shader
    highp vec3 normal = vertNormal;
    highp vec3 lightDir = normalize(lightPos - vert);
    highp vec3 viewDir = normalize(cameraPos - vert);

    highp float normalViewDirCosMax = sign(dot(viewDir, normal));
    highp vec3 color = normalViewDirCosMax * frontColor + (1 - normalViewDirCosMax) * backColor;

    fragColor = max(dot(normal, lightDir), 0.2) * vec4(color, 1.0);
}