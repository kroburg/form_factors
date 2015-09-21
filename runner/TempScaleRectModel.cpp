#include "TempScaleRectModel.h"

TempScaleRectModel::TempScaleRectModel() : vertices({ 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f }) {
}

TempScaleRectModel::~TempScaleRectModel() {
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &vao);
}

void TempScaleRectModel::init(int width, int height, int screenWidth, int screenHeight, const char* vertexShaderPath, const char* fragmentShaderPath) {
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

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void TempScaleRectModel::draw(int x, int y) {
    glDisable(GL_DEPTH_TEST);
    shader.bind();
    vec3 position((float)x, (float)y, 0.0f);
    mat4 model;
    model = translate(model, position);
    model = scale(model, vec3((float)width, (float)height, 1.0f));
    shader.setUniform("mvMatrix", model);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    shader.release();
}

void TempScaleRectModel::setSize(int width, int height, int screenWidth, int screenHeight) {
    this->width = width;
    this->height = height;
    this->screenWidth = screenWidth;
    this->screenHeight = screenHeight;

    shader.bind();
    mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f, -1.0f, 1.0f);
    shader.setUniform("projMatrix", projection);
}