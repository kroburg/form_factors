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

#include "OpenGLShaderProgram.h"

OpenGLShaderProgram::OpenGLShaderProgram(): vShaderHandle(0), fShaderHandle(0), shaderProgramHandle(0) {}

void OpenGLShaderProgram::init() {
    shaderProgramHandle = glCreateProgram();
}

OpenGLShaderProgram::~OpenGLShaderProgram() {
    if (shaderProgramHandle) {
        glUseProgram(0);
        if (vShaderHandle) {
            glDetachShader(shaderProgramHandle, vShaderHandle);
            glDeleteShader(vShaderHandle);
        }
        if (fShaderHandle) {
            glDetachShader(shaderProgramHandle, fShaderHandle);
            glDeleteShader(fShaderHandle);
        }    
        glDeleteProgram(shaderProgramHandle);
    }
}

static char* readFile(const char *filePath, size_t& length);

int OpenGLShaderProgram::addShaderFromSourceFile(ShaderType type, const char*path) {
    const char* name = type == OpenGLShaderProgram::VertexType ? "vertex" : "fragment";
    size_t len;
    char* shaderSource = readFile(path, len);
    if (!shaderSource) {
        ERROR("Unable to load " << name << " shader from \"" << path << "\"");
        return 1;
    }
    GLuint& shaderHandle = type == OpenGLShaderProgram::VertexType ? vShaderHandle : fShaderHandle;
    shaderHandle = glCreateShader(type == OpenGLShaderProgram::VertexType ? GL_VERTEX_SHADER : GL_FRAGMENT_SHADER);
    glShaderSource(shaderHandle, 1, (const GLchar**)&shaderSource, 0);
    glCompileShader(shaderHandle);
    GLint compiled = 0;
    glGetShaderiv(shaderHandle, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        ERROR("Unable to compile " << name << " shader from \"" << path << "\"");
        GLint maxLength = 0;
        glGetShaderiv(shaderHandle, GL_INFO_LOG_LENGTH, &maxLength);
        char* message = new char[maxLength];
        glGetShaderInfoLog(shaderHandle, maxLength, &maxLength, message);
        ERROR(message);
        delete[] message;
        delete[] shaderSource;
        return 1;
    }
    glAttachShader(shaderProgramHandle, shaderHandle);

    delete[] shaderSource;
    TRACE("Loaded " << name << " shader from \"" << path << "\"");

    return 0;
}

int OpenGLShaderProgram::link() {
    glLinkProgram(shaderProgramHandle);
    GLint linked;
    glGetProgramiv(shaderProgramHandle, GL_LINK_STATUS, &linked);
    if (!linked) {
        ERROR("Unable to link shader program");
        GLint maxLength = 0;
        glGetProgramiv(shaderProgramHandle, GL_INFO_LOG_LENGTH, &maxLength);
        char* message = new char[maxLength];
        glGetProgramInfoLog(shaderProgramHandle, maxLength, &maxLength, message);
        ERROR(message);
        delete[] message;
        return 1;
    }
    return 0;
}

int OpenGLShaderProgram::bindAttribLocation(const char* attrName, GLuint pos) {
    int maxAttrib;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxAttrib);
    if (pos >= maxAttrib) {
        return 1;
    }
    glBindAttribLocation(shaderProgramHandle, pos, attrName);
    return 0;
}

void OpenGLShaderProgram::bind() {
    glUseProgram(shaderProgramHandle);
}

void OpenGLShaderProgram::release() {
    glUseProgram(0);
}

std::vector<GLUniformInfo> OpenGLShaderProgram::getAllUniforms() {
    GLint count;
    glGetProgramiv(shaderProgramHandle, GL_ACTIVE_UNIFORMS, &count);
    int maxUniformNameLength;
    glGetProgramiv(shaderProgramHandle, GL_ACTIVE_UNIFORM_MAX_LENGTH, &maxUniformNameLength);
    char *name = new char[maxUniformNameLength];
    GLuint type = GL_NONE;
    GLint size = 0;
    GLsizei length = 0;
    std::vector<GLUniformInfo> result;
    for (GLuint i = 0; i < count; ++i) {
        glGetActiveUniform(shaderProgramHandle, i, maxUniformNameLength, &length, &size, &type, name);
        GLUniformInfo uniform;
        uniform.size = size;
        uniform.type = type;
        uniform.name = std::string(name);
        result.push_back(uniform);
    }
    delete[] name;
    return result;
}

std::vector<GLAttribInfo> OpenGLShaderProgram::getAllAttributes() {
    GLint count;
    glGetProgramiv(shaderProgramHandle, GL_ACTIVE_ATTRIBUTES, &count);
    int maxAttrNameLength;
    glGetProgramiv(shaderProgramHandle, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &maxAttrNameLength);
    char *name = new char[maxAttrNameLength];
    GLuint type = GL_NONE;
    GLint size = 0;
    GLsizei length = 0;
    std::vector<GLAttribInfo> result;
    for (GLuint i = 0; i < count; ++i) {
        glGetActiveAttrib(shaderProgramHandle, i, maxAttrNameLength, &length, &size, &type, name);
        GLAttribInfo attrib;
        attrib.size = size;
        attrib.type = type;
        attrib.name = std::string(name);
        attrib.index = getAttribLocation(name);
        result.push_back(attrib);
    }
    delete[] name;
    return result;
}

GLuint OpenGLShaderProgram::getAttribLocation(const char *attrName) const {
    return static_cast<GLuint>(glGetAttribLocation(shaderProgramHandle, attrName));
}

const char* OpenGLShaderProgram::getTypeName(GLuint type) {
    switch (type) {
        case GL_NONE:
            return "GL_NONE";
        case GL_FLOAT:
            return "GL_FLOAT";
        case GL_FLOAT_VEC2:
            return "GL_FLOAT_VEC2";
        case GL_FLOAT_VEC3:
            return "GL_FLOAT_VEC3";
        case GL_FLOAT_VEC4:
            return "GL_FLOAT_VEC4";
        case GL_INT:
            return "GL_INT";
        case GL_INT_VEC2:
            return "GL_INT_VEC2";
        case GL_INT_VEC3:
            return "GL_INT_VEC3";
        case GL_INT_VEC4:
            return "GL_INT_VEC4";
        case GL_BOOL:
            return "GL_BOOL";
        case GL_BOOL_VEC2:
            return "GL_BOOL_VEC2";
        case GL_BOOL_VEC3:
            return "GL_BOOL_VEC3";
        case GL_BOOL_VEC4:
            return "GL_BOOL_VEC4";
        case GL_FLOAT_MAT2:
            return "GL_FLOAT_MAT2";
        case GL_FLOAT_MAT3:
            return "GL_FLOAT_MAT3";
        case GL_FLOAT_MAT4:
            return "GL_FLOAT_MAT4";
        case GL_SAMPLER_1D:
            return "GL_SAMPLER_1D";
        case GL_SAMPLER_2D:
            return "GL_SAMPLER_2D";
        case GL_SAMPLER_3D:
            return "GL_SAMPLER_3D";
        case GL_SAMPLER_CUBE:
            return "GL_SAMPLER_CUBE";
        case GL_SAMPLER_1D_SHADOW:
            return "GL_SAMPLER_1D_SHADOW";
        case GL_SAMPLER_2D_SHADOW:
            return "GL_SAMPLER_2D_SHADOW";
        default:
            return "UNDEFINED";
    }
}

int OpenGLShaderProgram::setUniform(const char* name, float value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, float value) {
    glUniform1f(loc, value);
}

int OpenGLShaderProgram::setUniform(const char* name, glm::vec2& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::vec2& value) {
    glUniform2f(loc, value.x, value.y);
}

int OpenGLShaderProgram::setUniform(const char* name, glm::vec3& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::vec3& value) {
    glUniform3f(loc, value.x, value.y, value.z);
}

int OpenGLShaderProgram::setUniform(const char* name, glm::vec4& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::vec4& value) {
    glUniform4f(loc, value.x, value.y, value.z, value.w);
}

int OpenGLShaderProgram::setUniformArray(const char* name, GLsizei size, const float* array) {
    if (size <= 0) {
        return 1;
    }
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform1Array(loc, size, array);
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniformArray(const char* name, GLsizei size, const glm::vec2* array) {
    if (size <= 0) {
        return 1;
    }
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform2Array(loc, size, reinterpret_cast<const float*>(array));
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniformArray(const char* name, GLsizei size, const glm::vec3* array) {
    if (size <= 0) {
        return 1;
    }
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform3Array(loc, size, reinterpret_cast<const float*>(array));
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniformArray(const char* name, GLsizei size, const glm::vec4* array) {
    if (size <= 0) {
        return 1;
    }
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform4Array(loc, size, reinterpret_cast<const float*>(array));
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform1Array(GLint loc, GLsizei size, const float *array) {
    glUniform1fv(loc, size, array);
}

inline void OpenGLShaderProgram::setUniform2Array(GLint loc, GLsizei size, const float *array) {
    glUniform2fv(loc, size, array);
}

inline void OpenGLShaderProgram::setUniform3Array(GLint loc, GLsizei size, const float *array) {
    glUniform3fv(loc, size, array);
}

inline void OpenGLShaderProgram::setUniform4Array(GLint loc, GLsizei size, const float *array) {
    glUniform4fv(loc, size, array);
}

int OpenGLShaderProgram::setUniform(const char* name, int value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        glUniform1i(loc, value);
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniform(const char* name, glm::ivec2& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        glUniform2i(loc, value.x, value.y);
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniform(const char* name, glm::ivec3& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        glUniform3i(loc, value.x, value.y, value.z);
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniform(const char* name, glm::ivec4& value) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        glUniform4i(loc, value.x, value.y, value.z, value.w);
        return 0;
    }
    return 1;
}

int OpenGLShaderProgram::setUniform(const char* name, glm::mat2& value, bool transpose) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value, transpose);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::mat2& value, bool transpose) {
    glUniformMatrix2fv(loc, 1, static_cast<GLboolean>(transpose), glm::value_ptr(value));
}

int OpenGLShaderProgram::setUniform(const char* name, glm::mat3& value, bool transpose) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value, transpose);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::mat3& value, bool transpose) {
    glUniformMatrix3fv(loc, 1, static_cast<GLboolean>(transpose), glm::value_ptr(value));
}

int OpenGLShaderProgram::setUniform(const char* name, glm::mat4& value, bool transpose) {
    GLint loc = glGetUniformLocation(shaderProgramHandle, name);
    if (loc != -1) {
        setUniform(loc, value, transpose);
        return 0;
    }
    return 1;
}

inline void OpenGLShaderProgram::setUniform(GLint loc, glm::mat4& value, bool transpose) {
    glUniformMatrix4fv(loc, 1, static_cast<GLboolean>(transpose), glm::value_ptr(value));
}

char* OpenGLShaderProgram::readFile(const char *filePath, size_t& length) {
    FILE *fptr;
    fptr = fopen(filePath, "rb");
    if (!fptr) {
        length = 0;
        return nullptr;
    }
    fseek(fptr, 0, SEEK_END);
    length = (size_t)ftell(fptr);
    char *buf = new char[length + 1];
    fseek(fptr, 0, SEEK_SET);
    fread(buf, length, sizeof(char), fptr);
    fclose(fptr);
    buf[length] = 0;
    return buf;
}