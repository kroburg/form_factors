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
    playing(false),
    pos(0),
    cameraVec(vec3(0, 2, 5)) {}

ModelRenderer::~ModelRenderer() {
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
    // Init GL context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
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

    /*if (SDL_GL_SetSwapInterval(1) < 0) {
        logSDLError(std::cerr, "SDL_GL_SetSwapInterval");
        return 1;
    }*/

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

    TRACE("Init OpenGL buffers");
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glBufferData(GL_ARRAY_BUFFER, model->vSize() + model->nSize() + model->tSize(), 0, GL_DYNAMIC_COPY);
    glBufferSubData(GL_ARRAY_BUFFER, 0, model->vSize(), model->getVertices());
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize(), model->nSize(), model->getNormals());
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize() + model->nSize(), model->tSize(), model->getTemps());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<const void *>(model->vSize()));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, reinterpret_cast<const void *>(model->vSize() + model->nSize()));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, model->iSize(), model->getIndices(), GL_STATIC_DRAW);

    TRACE("OpenGL arrays and buffers done, setting global OpenGL parameters");

    glClearColor(0.0, 0.0, 0.0, 1.0);
    //glEnable(GL_CULL_FACE);

    TRACE("Uploading transformations to video card");

    sprogram.bind();
    // Vertex shader uniforms
    mvMatrix = glm::mat4(1.0);
    cameraMatrix = glm::lookAt(cameraVec, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f * width / height, 0.01f, 1000.0f);
    normalMatrix = glm::inverseTranspose(glm::mat3(mvMatrix));
    sprogram.setUniform("mvMatrix", mvMatrix);
    sprogram.setUniform("cameraMatrix", cameraMatrix);
    sprogram.setUniform("projMatrix", projectionMatrix);
    sprogram.setUniform("normalMatrix", normalMatrix);
    // Fragment shader uniforms
    lightPos = glm::vec3(0, 5, 10);
    cameraPos = -glm::vec3(cameraMatrix[3]) * glm::mat3(cameraMatrix);
    sprogram.setUniform("lightPos", lightPos);
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
    mvMatrix = glm::rotate(mvMatrix, radians(60.0f * update), glm::vec3(0, 1, 0));
    normalMatrix = glm::inverseTranspose(glm::mat3(mvMatrix));

    if (playing) {
        pos += update / tl.getCurTime();
        if (pos > 1.0f) {
            playing = false;
            pos = 1.0f;
        }
        vector<float> temps;
        tl.findTempForPos(pos, temps);
        colorModelForTemps(temps);
        timeLine.setPos(pos);
    }
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
    switch (event.type) {
        case (SDL_USEREVENT): {
            switch (event.user.code) {
                case (EV_NEWFRAME) : {
                    Timeline::frame_t* frame = (Timeline::frame_t *)event.user.data1;
                    assert(frame);
                    tl.addFrame(*frame);
                    timeLine.setPos(1.0);
                    colorModelForTemps(frame->temps);
                    delete frame;
                    break;
                }
                case (EV_PARSEDONE) : {
                    break;
                }
                default:
                    break;
            }
        }
        case (SDL_MOUSEWHEEL) : {
            if (event.wheel.y) {
                cameraVec += cameraVec * ((float)event.wheel.y / cameraVec.length() / cameraVec.length());
                cameraMatrix = glm::lookAt(cameraVec, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
                sprogram.bind();
                sprogram.setUniform("cameraMatrix", cameraMatrix);
            }
            break;
        }
        case (SDL_MOUSEBUTTONDOWN): {
            int x = event.button.x;
            int y = event.button.y;
            if (event.button.button == SDL_BUTTON_LEFT) {
                auto p = timeLine.getPosBy(x, y);
                if (p >= 0) {
                    pos = p;
                    vector<float> temps;
                    if (tl.findTempForPos(p, temps) == 0) {
                        colorModelForTemps(temps);
                    }
                    timeLine.setPos(p);
                }
            }
            break;
        }
        case (SDL_KEYDOWN) : {
            SDL_KeyboardEvent kev = event.key;
            if (kev.keysym.sym == SDLK_SPACE) {
                playing = !playing;
            }
            break;
        }
        default:
            break;
    }
}

void ModelRenderer::colorModelForTemps(vector<float>& temps) {
    // TODO smth with complexity
    for (std::size_t i = 0; i < temps.size(); ++i)
    {
        float temp = temps[i];
        subject::mesh_t mesh = scene->meshes[i];
        for (int j = mesh.first_idx; j < mesh.first_idx + mesh.n_faces; ++j)
        {
          int k = j * 3;
          model->temps[k] = temp;
          model->temps[k + 1] = temp;
          model->temps[k + 2] = temp;
        }
    }
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize() + model->nSize(), model->tSize(), model->getTemps());
}