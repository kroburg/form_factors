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

#pragma once

#include <vector>
#include <GL\glew.h>
#include <glm\glm.hpp>

using namespace std;
using namespace glm;

class Model {
public:
    vector<vec3> vertices;
    vector<vec3> normals;
    vector<int> indices;
    vector<float> temps;

    int vSize() const;
    int tSize() const;
    int nSize() const;
    int iSize() const;
    int iCount() const;

    GLfloat* getVertices() const;
    GLfloat* getNormals() const;
    GLfloat* getTemps() const;
    int* getIndices() const;

    Model(const vector<vec3>& vertices, const vector<vec3>& normals, const vector<int>& indices, const vector<float>& temps);
    ~Model();
};