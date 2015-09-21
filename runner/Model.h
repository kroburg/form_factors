#pragma once

#include <vector>
#include <GL\glew.h>
#include <glm\glm.hpp>

using namespace std;
using namespace glm;

class Model {
public:
    vector<vec3> vertices;
    vector<vec3> normals;
    vector<int> indices;
    vector<float> temps;

    int vSize() const;
    int tSize() const;
    int nSize() const;
    int iSize() const;
    int iCount() const;

    GLfloat* getVertices() const;
    GLfloat* getNormals() const;
    GLfloat* getTemps() const;
    int* getIndices() const;

    Model(const vector<vec3>& vertices, const vector<vec3>& normals, const vector<int>& indices, const vector<float>& temps);
    ~Model();
};