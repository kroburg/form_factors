// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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

#include "obj_import.h"
#include "../math/operations.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

namespace obj_import
{
  struct idx_face_t
  {
    unsigned int indices[3];
  };

  int import_obj(const char* filename, form_factors::scene_t** scene)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      return -OBJ_IMPORT_FILE_ERROR;
    }

    math::vec3 minBox = math::make_vec3(1e10, 1e10, 1e10);
    math::vec3 maxBox = math::make_vec3(-1e10, -1e10, -1e10);

    std::vector<math::vec3> vertices;
    std::vector<idx_face_t> faces;
    std::vector<form_factors::mesh_t> meshes;
    std::string str;

    while (getline(file, str))
    {
      if (str == "")
      {
        continue;
      }

      switch (str[0])
      {
      case 'v':
      {
                math::vec3 vertex;
                int count = sscanf(str.c_str() + 1, "%f %f %f", &(vertex.x), &(vertex.y), &(vertex.z));
                if (count == 3)
                {
                  vertices.push_back(vertex);
                  minBox = min(minBox, vertex);
                  maxBox = max(maxBox, vertex);
                }
                else
                {
                  return -OBJ_IMPORT_FORMAT_ERROR;
                }
      }
        break;

      case 'f':
      {
                int i0, i1, i2;
                i0 = i1 = i2 = 0;
                int count = sscanf(str.c_str() + 1, "%d %d %d", &i0, &i1, &i2);
                if (count == 3)
                {
                  idx_face_t face = { i0, i1, i2 };
                  faces.push_back(face);
                }
                else
                {
                  return -OBJ_IMPORT_FORMAT_ERROR;
                }
      }
        break;

      case 'm':
      {
                int start_idx, n_faces;
                int count = sscanf(str.c_str() + 1, "%d %d %d", &start_idx, &n_faces);
                if (count == 2)
                {
                  form_factors::mesh_t mesh = { start_idx, n_faces };
                  meshes.push_back(mesh);
                }
                else
                {
                  return -OBJ_IMPORT_FORMAT_ERROR;
                }
      }

      default:
        break;
      }
    }

    file.close();

    math::point_t cubeLen = std::max(std::max(maxBox.x - minBox.x, maxBox.y - minBox.y), maxBox.z - minBox.z);
    math::vec3 translation_vec = (maxBox + minBox) / 2;
    for (int i = 0, l = vertices.size(); i < l; ++i)
    {
      vertices[i] -= translation_vec;
      vertices[i] /= cubeLen;
    }

    form_factors::scene_t* result = form_factors::scene_create();
    *scene = result;

    result->n_faces = faces.size();
    result->faces = (form_factors::face_t *)malloc(result->n_faces * sizeof(form_factors::face_t));
    for (int i = 0; i < result->n_faces; ++i)
    {
      idx_face_t idx_face = faces[i];
      result->faces[i] = {
        vertices[idx_face.indices[0]],
        vertices[idx_face.indices[1]],
        vertices[idx_face.indices[2]]
      };
    }

    result->n_meshes = std::max(1, (int)meshes.size());
    result->meshes = (form_factors::mesh_t *)malloc(sizeof(form_factors::mesh_t) * result->n_meshes);
    if (meshes.empty())
    {
      result->meshes[0].first_idx = 0;
      result->n_faces = result->meshes[0].n_faces = faces.size();
    }
    else
    {
      memcpy(result->meshes, meshes.data(), sizeof(form_factors::mesh_t) * result->n_meshes);
    }

    return OBJ_IMPORT_OK;
  }
}
