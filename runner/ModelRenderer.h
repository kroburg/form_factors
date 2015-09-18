#pragma once;

#include "proj_defs.h"
#include "AppContainer.h"
#include "OpenGLShaderProgram.h"
#include "CubeGenerator.h"
#include <glm\\glm.hpp>
#include <glm\gtc\matrix_inverse.hpp>
#include <glm\gtc\matrix_transform.hpp>

class ModelRenderer: public AppContainer {
private:
    SDL_GLContext glContext;

    glm::mat4 mvMatrix;
    glm::mat4 cameraMatrix;
    glm::mat4 projectionMatrix;
    glm::mat3 normalMatrix;
    glm::vec3 lightPos;
    glm::vec3 cameraPos;
    glm::vec3 frontColor;
    glm::vec3 backColor;

    GLuint vao;
    GLuint vbo[2];

    OpenGLShaderProgram sprogram;
    Model* model;
    CubeGenerator cg;

public:
    virtual void onEvent(SDL_Event& event) override;
    virtual void onTick(float update) override;
    virtual void onRender() override;
    virtual int afterInit() override;
    virtual void onResize(int newWidth, int newHeight) override;

    ModelRenderer(const char* name);
    ~ModelRenderer();

    void logProgramParams() const;
};