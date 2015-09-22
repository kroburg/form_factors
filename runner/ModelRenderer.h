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

#pragma once;

#include <glm\glm.hpp>
#include <glm\gtx\quaternion.hpp>
#include <glm\gtc\matrix_inverse.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
#include "proj_defs.h"
#include "AppContainer.h"
#include "OpenGLShaderProgram.h"
#include "TempScaleRectModel.h"
#include "TimelineRectModel.h"
#include "CubeGenerator.h"
#include "Timeline.h"
#include "../import_export/obj_import.h"
#include "../subject/system.h"

#define JOY_DEAD_ZONE 4000

class ModelRenderer: public AppContainer {
private:
    SDL_GLContext glContext;

    glm::mat4 mvMatrix;
    glm::mat4 cameraMatrix;
    glm::mat4 projectionMatrix;
    glm::mat3 normalMatrix;
    glm::vec3 lightPos;
    glm::vec3 cameraPos;

    GLuint vao;
    GLuint vbo[2];

    OpenGLShaderProgram sprogram;
    Model* model;
    TempScaleRectModel tempScale;
    TimelineRectModel timeLine;
    Timeline tl;

    subject::scene_t* scene;

    void colorModelForTemps(vector<float>& temps);

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