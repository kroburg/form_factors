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

#include "CubeGenerator.h"

const vec3 CubeGenerator::gvertices[] = {
    vec3(-1, -1, 1),
    vec3(1, -1, 1),
    vec3(1, 1, 1),
        
    vec3(1, 1, 1),
    vec3(-1, 1, 1),
    vec3(-1, -1, 1),
        
    vec3(1, -1, 1),
    vec3(1, -1, -1),
    vec3(1, 1, -1),
        
    vec3(1, 1, -1),
    vec3(1, 1, 1),
    vec3(1, -1, 1),
        
    vec3(1, -1, -1),
    vec3(-1, -1, -1),
    vec3(-1, 1, -1),
        
    vec3(-1, 1, -1),
    vec3(1, 1, -1),
    vec3(1, -1, -1),
        
    vec3(-1, -1, -1),
    vec3(-1, -1, 1),
    vec3(-1, 1, 1),
        
    vec3(-1, 1, 1),
    vec3(-1, 1, -1),
    vec3(-1, -1, -1),
        
    vec3(1, -1, 1),
    vec3(-1, -1, 1),
    vec3(-1, -1, -1),
        
    vec3(-1, -1, -1),
    vec3(1, -1, -1),
    vec3(1, -1, 1),
        
    vec3(1, 1, -1),
    vec3(-1, 1, -1),
    vec3(-1, 1, 1),
        
    vec3(-1, 1, 1),
    vec3(1, 1, 1),
    vec3(1, 1, -1)
};

const vec3 CubeGenerator::gnormals[] = {
    vec3(0, 0, 1),
    vec3(0, 0, 1),
    vec3(0, 0, 1),

    vec3(0, 0, 1),
    vec3(0, 0, 1),
    vec3(0, 0, 1),

    vec3(1, 0, 0),
    vec3(1, 0, 0),
    vec3(1, 0, 0),

    vec3(1, 0, 0),
    vec3(1, 0, 0),
    vec3(1, 0, 0),

    vec3(0, 0, -1),
    vec3(0, 0, -1),
    vec3(0, 0, -1),

    vec3(0, 0, -1),
    vec3(0, 0, -1),
    vec3(0, 0, -1),

    vec3(-1, 0, 0),
    vec3(-1, 0, 0),
    vec3(-1, 0, 0),

    vec3(-1, 0, 0),
    vec3(-1, 0, 0),
    vec3(-1, 0, 0),

    vec3(0, -1, 0),
    vec3(0, -1, 0),
    vec3(0, -1, 0),

    vec3(0, -1, 0),
    vec3(0, -1, 0),
    vec3(0, -1, 0),

    vec3(0, 1, 0),
    vec3(0, 1, 0),
    vec3(0, 1, 0),

    vec3(0, 1, 0),
    vec3(0, 1, 0),
    vec3(0, 1, 0)
};

Model* CubeGenerator::next(const vec3& scaleVec, const mat4& transform) {
    // Transform matrix with per-axis scale factor
    mat4 A = transform * glm::diagonal4x4(vec4(scaleVec, 1.0));
    // Matrix for normals rotation (but not translation)
    mat3 B = mat3(transform);
    vector<vec3> vertices, normals;
    vector<int> indices;
    for (int i = 0; i < 36; ++i) {
        auto vec = vec3(A * vec4(gvertices[i], 1.0));
        vertices.push_back(vec);
    }
    for (int i = 0; i < 36; ++i) {
        auto vec = B * gnormals[i];
        normals.push_back(vec);
    }
    for (int i = 0; i < 36; ++i) {
        indices.push_back(i);
    }
    vector<float> temps(vertices.size());
    fill(temps.begin(), temps.end(), 273.0f);
    return new Model(vertices, normals, indices, temps);
}

Model* CubeGenerator::next(Model* toMergeWith, const vec3& scaleVec, const mat4& transform) {
    int lastIdx = toMergeWith->indices[toMergeWith->indices.size() - 1] + 1;
    for (int i = 0; i < 36; ++i) {
        toMergeWith->indices.push_back(i + lastIdx);
    }
    // Transform matrix with per-axis scale factor
    mat4 A = transform * glm::diagonal4x4(vec4(scaleVec, 1.0));
    // Matrix for normals rotation (but not translation)
    mat3 B = mat3(transform);
    vector<vec3> vertices, normals;
    vector<int> indices;
    for (int i = 0; i < 36; ++i) {
        auto vec = vec3(A * vec4(gvertices[i], 1.0));
        toMergeWith->vertices.push_back(vec);
        toMergeWith->temps.push_back(273.0f);
    }
    for (int i = 0; i < 36; ++i) {
        auto vec = B * gnormals[i];
        toMergeWith->normals.push_back(vec);
    }
    return toMergeWith;
}