#include "ModelRenderer.h"

using namespace utils;

ModelRenderer::ModelRenderer(const char* name, subject::scene_t* scene) : AppContainer(name), model(NULL), glContext(NULL), scene(scene) { }

ModelRenderer::~ModelRenderer() {
    if (model) {
        delete model;
    }
    if (glContext) {
        SDL_GL_DeleteContext(glContext);
    }
}

int ModelRenderer::afterInit() {
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
        sprogram.bindAttribLocation("vertex", 0);
        sprogram.bindAttribLocation("normal", 1);
        if (sprogram.link() != 0) {
            ERROR("Can not link program");
            return 1;
        }
    }

    if (!scene) {
        TRACE("Shaders done, creating model (no scene set)");
        model = cg.next();
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(1, 0, 0)));
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(-1, 0, 0)));
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(0, 0, 1)));
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(0, 0, -1)));
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(0, 1, 0)));
        cg.next(model, vec3(0.5), glm::translate(mat4(1.0f), vec3(0, -1, 0)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(3, 0, 0)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(-3, 0, 0)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(0, 0, 3)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(0, 0, -3)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(0, 3, 0)));
        cg.next(model, vec3(0.25), glm::translate(mat4(1.0f), vec3(0, -3, 0)));
        cg.next(model, vec3(1, 0.125, 0.125), glm::translate(mat4(1.0f), vec3(2, 0, 0)));
        cg.next(model, vec3(1, 0.125, 0.125), glm::translate(mat4(1.0f), vec3(-2, 0, 0)));
        cg.next(model, vec3(0.125, 0.125, 1), glm::translate(mat4(1.0f), vec3(0, 0, 2)));
        cg.next(model, vec3(0.125, 0.125, 1), glm::translate(mat4(1.0f), vec3(0, 0, -2)));
        cg.next(model, vec3(0.125, 1, 0.125), glm::translate(mat4(1.0f), vec3(0, 2, 0)));
        cg.next(model, vec3(0.125, 1, 0.125), glm::translate(mat4(1.0f), vec3(0, -2, 0)));
        TRACE("Model created, preparing OpengGL arrays and buffers");
    }
    else {
        TRACE("Extracting model from scene");
        std::vector<vec3> vert;
        std::vector<vec3> normals;
        std::vector<int> indices(scene->n_faces);
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 0; i < scene->n_faces; ++i) {
            vec3 a = vec3(scene->faces[i].points[0].x, scene->faces[i].points[0].y, scene->faces[i].points[0].z);
            vec3 b = vec3(scene->faces[i].points[1].x, scene->faces[i].points[1].y, scene->faces[i].points[1].z);
            vec3 c = vec3(scene->faces[i].points[2].x, scene->faces[i].points[2].y, scene->faces[i].points[2].z);
            vert.push_back(a);
            vert.push_back(b);
            vert.push_back(c);
            normals.push_back(cross(a - b, a - c));
            normals.push_back(cross(b - a, b - c));
            normals.push_back(cross(c - a, c - b));
        }
        model = new Model(vert, normals, indices);
    }

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(2, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo[1]);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBufferData(GL_ARRAY_BUFFER, model->vSize() * 2, model->getVertices(), GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, model->vSize(), model->nSize(), model->getNormals());
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), reinterpret_cast<const void *>(model->vSize()));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, model->iSize(), model->getIndices(), GL_STATIC_DRAW);

    TRACE("OpenGL arrays and buffers done, setting global OpenGL parameters");

    glClearColor(0.0, 0.0, 0.0, 1.0);
    //glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    TRACE("Uploading transformations to video card");

    sprogram.bind();
    // Vertex shader uniforms
    mvMatrix = glm::mat4(1.0);
    cameraMatrix = glm::lookAt(glm::vec3(0, 5, 10), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
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
    sprogram.release();

    TRACE("All initializaiton done");

    return 0;
}

void ModelRenderer::onRender() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    sprogram.bind();
    sprogram.setUniform("mvMatrix", mvMatrix);
    sprogram.setUniform("projMatrix", projectionMatrix);
    sprogram.setUniform("normalMatrix", normalMatrix);
    glDrawElements(GL_TRIANGLES, model->iCount(), GL_UNSIGNED_INT, 0);
    sprogram.release();

    SDL_GL_SwapWindow(window);
}

void ModelRenderer::onTick(float update) {
    mvMatrix = glm::rotate(mvMatrix, glm::radians(update / 16.0f), glm::vec3(0, 1, 0));
    normalMatrix = glm::inverseTranspose(glm::mat3(mvMatrix));
}

void ModelRenderer::onResize(int newWidth, int newHeight) {
    AppContainer::onResize(newWidth, newHeight);
    printf("Resized to %i x %i\n", width, height);
    projectionMatrix = glm::perspective(glm::radians(45.0f), 1.0f * width / height, 0.01f, 100.0f);
    glViewport(0, 0, width, height);
}

void ModelRenderer::onEvent(SDL_Event &event) {

}