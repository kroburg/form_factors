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

#include "ModelRenderer.h"

using namespace utils;

ModelRenderer::ModelRenderer(const char* name, subject::scene_t* scene) :
    AppContainer(name),
    model(NULL),
    glContext(NULL),
    scene(scene),
    totalFramesTime(0.0f),
    joy(NULL),
    joyX(0),
    joyY(0) { }

ModelRenderer::~ModelRenderer() {
    if (joy) {
        SDL_JoystickClose(joy);
        joy = NULL;
    }
    if (model) {
        delete model;
        model = NULL;
    }
    if (glContext) {
        SDL_GL_DeleteContext(glContext);
        glContext = NULL;
    }
}

int ModelRenderer::afterInit() {
    if (SDL_NumJoysticks() > 0) {
        joy = SDL_JoystickOpen(0);
    }

    // Init GL context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);

    glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        logSDLError(std::cerr, "SDL_GL_CreateContext");
        return 1;
    }

    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if (glewError != GLEW_OK) {
        ERROR("Error initializing GLEW: " << glewGetErrorString(glewError));
        return 1;
    }

    if (SDL_GL_SetSwapInterval(1) < 0) {
        logSDLError(std::cerr, "SDL_GL_SetSwapInterval");
        return 1;
    }

    TRACE("Initializing shaders");

    sprogram.init();
    if (sprogram.addShaderFromSourceFile(OpenGLShaderProgram::VertexType, "../shaders/mvp.vert") != 0 ||
        sprogram.addShaderFromSourceFile(OpenGLShaderProgram::FragmentType, "../shaders/simple.frag") != 0) {
        ERROR("Can not compile shaders");
        return 1;
    } else {
        if (sprogram.link() != 0) {
            ERROR("Can not link program");
            return 1;
        }
    }

    TRACE("Extracting model from scene");
    std::vector<vec3> vert;
    std::vector<vec3> normals;

    for (int i = 0; i < scene->n_faces; ++i) {
        vec3 a = vec3(scene->faces[i].points[0].x, scene->faces[i].points[0].y, scene->faces[i].points[0].z);
        vec3 b = vec3(scene->faces[i].points[1].x, scene->faces[i].points[1].y, scene->faces[i].points[1].z);
        vec3 c = vec3(scene->faces[i].points[2].x, scene->faces[i].points[2].y, scene->faces[i].points[2].z);
        vert.push_back(a);
        vert.push_back(b);
        vert.push_back(c);
        normals.push_back(cross(a - b, a - c));
        normals.push_back(cross(b - c, b - a));
        normals.push_back(cross(c - a, c - b));
    }

    std::vector<int> indices(vert.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<float> temps(indices.size());
    std::fill(temps.begin(), temps.end(), 0.0f);
    model = new Model(vert, normals, indices, temps);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBufferData(GL_ARRAY_BUFFER, model->vSize() + model->nSize() + model->tSize(), model->getVertices(), GL_DYNAMIC_COPY);
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize(), model->nSize(), model->getNormals());
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize() + model->nSize(), model->tSize(), model->getTemps());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), reinterpret_cast<const void *>(model->vSize()));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), reinterpret_cast<const void *>(model->vSize() + model->nSize()));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, model->iSize(), model->getIndices(), GL_STATIC_DRAW);

    TRACE("OpenGL arrays and buffers done, setting global OpenGL parameters");

    glClearColor(0.0, 0.0, 0.0, 1.0);
    //glEnable(GL_CULL_FACE);

    TRACE("Uploading transformations to video card");

    sprogram.bind();
    // Vertex shader uniforms
    mvMatrix = glm::mat4(1.0);
    cameraMatrix = glm::lookAt(glm::vec3(0, 2, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f * width / height, 0.01f, 100.0f);
    normalMatrix = glm::inverseTranspose(glm::mat3(mvMatrix));
    sprogram.setUniform("mvMatrix", mvMatrix);
    sprogram.setUniform("cameraMatrix", cameraMatrix);
    sprogram.setUniform("projMatrix", projectionMatrix);
    sprogram.setUniform("normalMatrix", normalMatrix);
    // Fragment shader uniforms
    lightPos = glm::vec3(0, 5, 10);
    cameraPos = -glm::vec3(cameraMatrix[3]) * glm::mat3(cameraMatrix);
    frontColor = glm::vec3(1, 0, 0);
    backColor = glm::vec3(0, 0, 1);
    sprogram.setUniform("lightPos", lightPos);
    sprogram.setUniform("frontColor", frontColor);
    sprogram.setUniform("backColor", backColor);
    sprogram.setUniform("cameraPos", cameraPos);
    // Temperatures
    sprogram.setUniform("maxTemp", 300.0f);

    tempScale.init(10, height, width, height, "../shaders/mvpRect.vert", "../shaders/fragRect.frag");
    timeLine.init(width - 30, 10, width, height, "../shaders/mvpRect.vert", "../shaders/fragRectPos.frag");

    TRACE("All initializaiton done");

    return 0;
}

void ModelRenderer::onRender() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    sprogram.bind();
    sprogram.setUniform("mvMatrix", mvMatrix);
    sprogram.setUniform("projMatrix", projectionMatrix);
    sprogram.setUniform("normalMatrix", normalMatrix);
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, model->iCount(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    tempScale.draw(width - 10, 0);
    timeLine.draw(10, height - 20);

    SDL_GL_SwapWindow(window);
}

void ModelRenderer::onTick(float update) {
    auto q = quat(vec3(joyY * 0.1f, joyX * 0.1f, 0));
    mvMatrix = glm::toMat4(q) * mvMatrix;
    //mvMatrix = glm::rotate(mvMatrix, joyY * 0.1f, glm::vec3(0, 1, 0));
    normalMatrix = glm::inverseTranspose(glm::mat3(mvMatrix));
}

void ModelRenderer::onResize(int newWidth, int newHeight) {
    AppContainer::onResize(newWidth, newHeight);
    printf("Resized to %i x %i\n", width, height);
    projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f * width / height, 0.01f, 100.0f);
    glViewport(0, 0, width, height);
    tempScale.setSize(10, height, width, height);
    timeLine.setSize(width - 30, 10, width, height);
}

void ModelRenderer::onEvent(SDL_Event &event) {
    static float curTime = 0.0f;
    switch (event.type) {
        case (SDL_USEREVENT): {
            switch (event.user.code) {
                case (EV_NEWFRAME) : {
                    frame_t* frame = (frame_t *)event.user.data1;
                    assert(frame);
                    curTime += frame->step;
                    // TODO smth with complexity
                    for (auto i = 0; i < std::min(frame->n_temps, scene->n_meshes); ++i) {
                        auto mesh = scene->meshes[i];
                        float curTemp = frame->temps[i];
                        for (auto j = mesh.first_idx; j < std::min(mesh.first_idx + mesh.n_faces, scene->n_faces); ++j) {
                            auto face = scene->faces[j];
                            for (int m = 0; m < 3; ++m) {
                                math::vec3 facevec = face.points[m];
                                for (int k = 0; k < model->vertices.size(); ++k) {
                                    vec3 vertex = model->vertices[k];
                                    if (fabsf(vertex.x - facevec.x) && fabsf(vertex.y - facevec.y) < EPS && fabsf(vertex.z - facevec.z) < EPS) {
                                        model->temps[k] = curTemp;
                                    }
                                }
                            }
                        }
                    }
                    glBufferSubData(GL_ARRAY_BUFFER, model->vSize() + model->nSize(), model->tSize(), model->getTemps());
                    delete frame;
                    break;
                }
                case (EV_PARSEDONE) : {
                    int result = (int)event.user.data1;
                    totalFramesTime = curTime;
                    curTime = 0.0f;
                    // TODO smth
                    break;
                }
                default:
                    break;
            }
        }
        case (SDL_JOYAXISMOTION): {
            if (event.jaxis.axis == 0) {
                joyX = abs((int)event.jaxis.value) < JOY_DEAD_ZONE ? 0.0 : (float)event.jaxis.value / 32768.0f;
            } else if (event.jaxis.axis == 1) {
                joyY = abs((int)event.jaxis.value) < JOY_DEAD_ZONE ? 0.0 : -(float)event.jaxis.value / 32768.0f;
            }
            break;
        }
        case (SDL_MOUSEBUTTONDOWN): {
            int x = event.button.x;
            int y = event.button.y;
            if (event.button.button == SDL_BUTTON_LEFT) {
                auto p = timeLine.getPosBy(x, y);
                if (p >= 0) {
                    timeLine.setPos(p);
                }
            }
        }
        default:
            break;
    }
}