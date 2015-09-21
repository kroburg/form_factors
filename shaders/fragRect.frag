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