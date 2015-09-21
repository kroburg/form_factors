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

int Model::tSize() const {
    return temps.size() * sizeof(float);
}

GLfloat* Model::getVertices() const {
    return const_cast<GLfloat *>((GLfloat *)vertices.data());
}

GLfloat* Model::getTemps() const {
    return const_cast<GLfloat *>((GLfloat *)temps.data());
}

GLfloat* Model::getNormals() const {
    return const_cast<GLfloat *>((GLfloat *)normals.data());
}

int* Model::getIndices() const {
    return const_cast<int *>(indices.data());
}

Model::Model(const vector<vec3>& vertices, const vector<vec3>& normals, const vector<int>& indices, const vector<float>& temps) :
    vertices(vertices), normals(normals), indices(indices), temps(temps) { }

Model::~Model() {
    temps.clear();
    vertices.clear();
    normals.clear();
    indices.clear();
}