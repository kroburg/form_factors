#version 150

in highp vec3 vert;
in highp vec3 vertNormal;
in highp float vertTemp;

uniform highp vec3 lightPos;
uniform highp vec3 frontColor;
uniform highp vec3 backColor;
uniform highp vec3 cameraPos;
uniform highp float maxTemp;

out highp vec4 fragColor;

void main() {
    // This is simple diffuse shader
    highp vec3 normal = vertNormal;
    highp vec3 lightDir = normalize(lightPos - vert);
    highp vec3 viewDir = normalize(cameraPos - vert);

    /*
    // Coloring faces with front or back color
    highp float normalViewDirCosMax = sign(dot(viewDir, normal));
    highp vec3 color = normalViewDirCosMax * frontColor + (1 - normalViewDirCosMax) * backColor;
    */

    float intrp = min(vertTemp / maxTemp, 1.0);
    vec3 lc = vec3(0, 0, 1);
    vec3 mc = vec3(0, 1, 0);
    vec3 hc = vec3(1, 0, 0);

    highp vec3 color;
    if (intrp < 0.5) {
        color = mix(lc, mc, intrp * 2);
    } else {
        color = mix(mc, hc, (intrp - 0.5) * 2);
    }

    fragColor = max(dot(normal, lightDir), 0.2) * vec4(color, 1.0);
}