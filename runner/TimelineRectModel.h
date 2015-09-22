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

#include <stdlib.h>
#include <array>
#include <stdexcept>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "OpenGLShaderProgram.h"

using namespace glm;

class TimelineRectModel {
private:
    int screenWidth;
    int screenHeight;
    int width;
    int height;
    int x;
    int y;

    GLuint vao;
    GLuint vbo;
    std::array<GLfloat, 12> vertices;
    OpenGLShaderProgram shader;
    mat4 projection;

    float pos;

public:
    TimelineRectModel();
    ~TimelineRectModel();

    void init(int width, int height, int screenWidth, int screenHeight, const char* vertexShaderPath, const char* fragmentShaderPath);
    void draw(int x, int y);
    void setSize(int width, int height, int screenWidth, int screenHeight);
    void setPos(float pos);

    float getPosBy(int x, int y);
};