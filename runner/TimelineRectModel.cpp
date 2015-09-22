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

#include "TimelineRectModel.h"

TimelineRectModel::TimelineRectModel() : vertices({ 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f }), pos(0.0f), x(0), y(0) {
}

TimelineRectModel::~TimelineRectModel() {
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
}

void TimelineRectModel::init(int width, int height, int screenWidth, int screenHeight, const char* vertexShaderPath, const char* fragmentShaderPath) {
    this->width = width;
    this->height = height;
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;

    shader.init();
    if (shader.addShaderFromSourceFile(OpenGLShaderProgram::VertexType, vertexShaderPath) != 0 ||
        shader.addShaderFromSourceFile(OpenGLShaderProgram::FragmentType, fragmentShaderPath)) {
        throw new std::runtime_error("Can not load/compile shaders");
    }
    else if (shader.link() != 0) {
        throw new std::runtime_error("Can not link program");
    }

    shader.bind();
    mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f, -1.0f, 1.0f);
    shader.setUniform("projMatrix", projection);
    shader.setUniform("pos", pos);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void TimelineRectModel::draw(int x, int y) {
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);
    shader.bind();
    this->x = x;
    this->y = y;
    vec3 position((float)x, (float)y, 0.0f);
    mat4 model;
    model = translate(model, position);
    model = scale(model, vec3((float)width, (float)height, 1.0f));
    shader.setUniform("mvMatrix", model);
    shader.setUniform("pos", pos);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    shader.release();
    glDisable(GL_BLEND);
}

void TimelineRectModel::setSize(int width, int height, int screenWidth, int screenHeight) {
    this->width = width;
    this->height = height;
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;

    shader.bind();
    mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f, -1.0f, 1.0f);
    shader.setUniform("projMatrix", projection);
}

void TimelineRectModel::setPos(float pos) {
    this->pos = pos;
}

float TimelineRectModel::getPosBy(int x, int y) {
    if (x >= this->x && y >= this->y && x < this->x + width && y < this->y + height) {
        return (float)(x - this->x) / width;
    } else {
        return -1.0f;
    }
}