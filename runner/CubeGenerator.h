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
#include <array>
#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtx\matrix_operation.hpp>
#include "Model.h"

using namespace std;
using namespace glm;

class CubeGenerator {
private:
    static const vec3 gvertices[36];
    static const vec3 gnormals[36];

public:
    Model* next(const vec3& scaleVec = vec3(1.0f), const mat4& transform = mat4(1.0f));
    Model* next(Model* toMergeWith, const vec3& scaleVec = vec3(1.0f), const mat4& transform = mat4(1.0f));
};