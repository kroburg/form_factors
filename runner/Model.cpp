#include "Model.h"

int Model::vSize() const {
    return vertices.size() * 3 * sizeof(GLfloat);
}

int Model::nSize() const {
    return normals.size() * 3 * sizeof(GLfloat);
}

int Model::iSize() const {
    return indices.size() * sizeof(int);;
}

int Model::iCount() const {
    return indices.size();
}

GLfloat* Model::getVertices() const {
    return const_cast<GLfloat *>((GLfloat *)vertices.data());
}

GLfloat* Model::getNormals() const {
    return const_cast<GLfloat *>((GLfloat *)normals.data());
}

int* Model::getIndices() const {
    return const_cast<int *>(indices.data());
}

Model::Model(const vector<vec3>& vertices, const vector<vec3>& normals, const vector<int>& indices):
    vertices(vertices), normals(normals), indices(indices) { }

Model::~Model() {
    vertices.clear();
    normals.clear();
    indices.clear();
}