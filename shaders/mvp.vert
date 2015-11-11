// Copyright (c) 2015 Contributors as noted in the AUTHORS file.
// This file is part of form_factors.
//
// form_factors is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// form_factors is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#version 150

in vec3 vertex;
in vec3 normal;
in float temp;

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