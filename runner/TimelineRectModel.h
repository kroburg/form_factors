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