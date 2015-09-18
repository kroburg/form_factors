#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "proj_defs.h"

struct GLUniformInfo {
    GLuint type;
    GLuint size;
    std::string name;
};

struct GLAttribInfo {
    GLint index;
    GLuint type;
    GLint size;
    std::string name;
};

class OpenGLShaderProgram {
private:
    GLuint vShaderHandle;
    GLuint fShaderHandle;
    GLuint shaderProgramHandle;

public:
    OpenGLShaderProgram();
    ~OpenGLShaderProgram();

    typedef int ShaderType;
    static const ShaderType VertexType = 1;
    static const ShaderType FragmentType = 2;

    int addShaderFromSourceFile(ShaderType type, const char* path);
    int bindAttribLocation(const char* attrName, GLuint pos);

    std::vector<GLUniformInfo> getAllUniforms();
    std::vector<GLAttribInfo> getAllAttributes();
    GLuint getAttribLocation(const char* attrName) const;

    const char* getTypeName(GLuint type);

    int setUniform(const char* name, float value);
    int setUniform(const char* name, glm::vec2& value);
    int setUniform(const char* name, glm::vec3& value);
    int setUniform(const char* name, glm::vec4& value);

    int setUniform(const char* name, int value);
    int setUniform(const char* name, glm::ivec2& value);
    int setUniform(const char* name, glm::ivec3& value);
    int setUniform(const char* name, glm::ivec4& value);

    int setUniform(const char* name, glm::mat2& value, bool transpose = false);
    int setUniform(const char* name, glm::mat3& value, bool transpose = false);
    int setUniform(const char* name, glm::mat4& value, bool transpose = false);

    int setUniformArray(const char* name, GLsizei size, const float* array);
    int setUniformArray(const char* name, GLsizei size, const glm::vec2* array);
    int setUniformArray(const char* name, GLsizei size, const glm::vec3* array);
    int setUniformArray(const char* name, GLsizei size, const glm::vec4* array);

    inline void setUniform(GLint loc, float value);
    inline void setUniform(GLint loc, glm::vec2& value);
    inline void setUniform(GLint loc, glm::vec3& value);
    inline void setUniform(GLint loc, glm::vec4& value);

    inline void setUniform(GLint loc, int value);
    inline void setUniform(GLint loc, glm::ivec2& value);
    inline void setUniform(GLint loc, glm::ivec3& value);
    inline void setUniform(GLint loc, glm::ivec4& value);

    inline void setUniform(GLint loc, glm::mat2& value, bool transpose = false);
    inline void setUniform(GLint loc, glm::mat3& value, bool transpose = false);
    inline void setUniform(GLint loc, glm::mat4& value, bool transpose = false);

    inline void setUniform1Array(GLint loc, GLsizei size, const float *array);
    inline void setUniform2Array(GLint loc, GLsizei size, const float *array);
    inline void setUniform3Array(GLint loc, GLsizei size, const float *array);
    inline void setUniform4Array(GLint loc, GLsizei size, const float *array);

    int link();
    void init();
    void bind();
    void release();

    static char* readFile(const char *filePath, size_t& length);

};