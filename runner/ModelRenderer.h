#pragma once;

#include <glm\glm.hpp>
#include <glm\gtx\quaternion.hpp>
#include <glm\gtc\matrix_inverse.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <vector>
#include <numeric>
#include "proj_defs.h"
#include "AppContainer.h"
#include "OpenGLShaderProgram.h"
#include "TempScaleRectModel.h"
#include "TimelineRectModel.h"
#include "CubeGenerator.h"
#include "../import_export/obj_import.h"
#include "../subject/system.h"

#define JOY_DEAD_ZONE 4000

struct frame_t {
    float step;
    float* temps;
    int n_temps;

    frame_t(float step, float* temps, int n_temps) : step(step), n_temps(n_temps) {
        this->temps = new float[n_temps];
        memcpy(this->temps, temps, n_temps * sizeof(float));
    }

    ~frame_t() { delete[] temps; }
};

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
    TempScaleRectModel tempScale;
    TimelineRectModel timeLine;

    SDL_Joystick *joy;
    float joyX;
    float joyY;

    subject::scene_t* scene;
    float totalFramesTime;

public:
    virtual void onEvent(SDL_Event& event) override;
    virtual void onTick(float update) override;
    virtual void onRender() override;
    virtual int afterInit() override;
    virtual void onResize(int newWidth, int newHeight) override;

    ModelRenderer(const char* name, subject::scene_t* scene = NULL);
    ~ModelRenderer();

    void logProgramParams() const;

    static const Sint32 EV_NEWFRAME = 1;
    static const Sint32 EV_PARSEDONE = 2;
};