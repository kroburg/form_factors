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