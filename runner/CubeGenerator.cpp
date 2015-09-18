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
    return new Model(vertices, normals, indices);
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
    }
    for (int i = 0; i < 36; ++i) {
        auto vec = B * gnormals[i];
        toMergeWith->normals.push_back(vec);
    }
    return toMergeWith;
}