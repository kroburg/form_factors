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

/**
 * This module contains functionality of loading obj-files (Wavefront).
 */

#include "obj_import.h"
#include "../math/operations.h"
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>

#ifdef _WIN32
#pragma warning(disable:4996)
#endif

namespace obj_import
{
  /// @brief Local structure representing vertex indices for given face (polygon).
  struct idx_face_t
  {
    int indices[3];
  };

  /// @note Mindflow mode on
  int scene(const char* filename, subject::scene_t** scene)
  {
    std::ifstream file(filename);
    if (!file.is_open())
    {
      return -OBJ_IMPORT_FILE_ERROR;
    }

    std::vector<math::vec3> vertices;
    std::vector<idx_face_t> faces;
    int mesh_start_idx = -1;
    std::vector<subject::mesh_t> meshes;
    std::vector<subject::material_t> materials;
    typedef std::map<std::string, std::size_t> MaterialMap;
    MaterialMap material2IndexMap;
    std::string str;
    unsigned line = 0;

    while (getline(file, str))
    {  
      ++line;
      if (str == "" || str[0] == '#')
      {
        continue;
      }

      reparse_line:
      bool retry_current_line = false;

      switch (str[0])
      {
      case 'v':
      {
        math::vec3 vertex;
        int count = sscanf(str.c_str() + 1, "%f %f %f", &(vertex.x), &(vertex.y), &(vertex.z));
        if (count == 3)
        {
          vertices.push_back(vertex);
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
          idx_face_t face = { i0 - 1, i1 - 1, i2 - 1 };
          faces.push_back(face);
        }
        else
        {
          return -OBJ_IMPORT_FORMAT_ERROR;
        }
      }
      break;

      case 'g':
      {
        subject::mesh_t mesh;
        mesh.first_idx = faces.size();
        mesh.material_idx = 0;
        if (!meshes.empty())
          meshes.back().n_faces = faces.size() - meshes.back().first_idx;
        meshes.push_back(mesh);
      }
      break;

      case 'u':
      {
        std::string::size_type pos = str.find_first_of(" \t");
        if (pos == std::string::npos)
        {
          fprintf(stderr, "Failed to parse line %d '%s'\n", line, str.c_str());
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string token = str.substr(0, pos);
        if (token != "usemtl")
        {
          fprintf(stderr, "Invalid material token at line %d '%s'\n", line, str.c_str());
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string name = str.substr(pos + 1);
        MaterialMap::const_iterator found = material2IndexMap.find(name);
        if (found == material2IndexMap.end())
        {
          fprintf(stderr, "Failed to find material reference '%s' at line %d\n", name.c_str(), line);
          return -OBJ_IMPORT_MATERIAL_NOT_DEFINED;
        }

        meshes.back().material_idx = found->second;
      }
      break;

      case 'n':
      {
        if (strncmp("newmtl", str.c_str(), 6) != 0)
        {
          fprintf(stderr, "Failed to parse newmtl at line %d\n", line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        std::string name = str.substr(7);

        subject::material_t m = subject::material_t();

        if (name.length() >= sizeof(m.name))
        {
          fprintf(stderr, "Material name '%s' is too long at line %d\n", name.c_str(), line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        strncpy(m.name, name.c_str(), std::min(sizeof(m.name), name.length()));

        typedef std::map<std::string, float*> MaterialPropertiesMap;
        MaterialPropertiesMap materialProperties;
        materialProperties["shell.density"] = &m.shell.density;
        materialProperties["shell.heat_capacity"] = &m.shell.heat_capacity;
        materialProperties["shell.thermal_conductivity"] = &m.shell.thermal_conductivity;
        materialProperties["shell.thickness"] = &m.shell.thickness;
        materialProperties["front.specular_reflectance"] = &m.front.specular_reflectance;
        materialProperties["front.diffuse_reflectance"] = &m.front.diffuse_reflectance;
        materialProperties["front.absorbance"] = &m.front.absorbance;
        materialProperties["front.transmittance"] = &m.front.transmittance;
        materialProperties["front.emissivity"] = &m.front.emissivity;
        materialProperties["rear.specular_reflectance"] = &m.rear.specular_reflectance;
        materialProperties["rear.diffuse_reflectance"] = &m.rear.diffuse_reflectance;
        materialProperties["rear.absorbance"] = &m.rear.absorbance;
        materialProperties["rear.transmittance"] = &m.rear.transmittance;
        materialProperties["rear.emissivity"] = &m.rear.emissivity;

        typedef std::set<std::string> StringSet;
        StringSet parsedParameters;

        while (getline(file, str))
        {
          ++line;
          if (str == "" || str[0] == '#')
            continue;

          std::string::size_type pos = str.find_first_of(" \t");
          if (pos + 1 >= str.length())
          {
            fprintf(stderr, "Invalid material definition format at line %d\n", line);
            return -OBJ_IMPORT_FORMAT_ERROR;
          }

          std::string propertyName = str.substr(0, pos);
          if (propertyName == "v" ||
            propertyName == "f" ||
            propertyName == "g")
          {
            retry_current_line = true;
            goto finish_material_parsing;
          }

          std::string propertyValue = str.substr(pos + 1);

          MaterialPropertiesMap::const_iterator found = materialProperties.find(propertyName);
          if (found == materialProperties.end())
          {
            fprintf(stderr, "Unknown material property '%s' at line %d\n", propertyName.c_str(), line);
            return -OBJ_IMPORT_FORMAT_ERROR;
          }

          if (sscanf(propertyValue.c_str(), "%f", found->second) != 1)
          {
            fprintf(stderr, "Failed to parse material '%s' value '%s' at line %d\n", propertyName.c_str(), propertyValue.c_str(), line);
            return -OBJ_IMPORT_MATERIAL_INVALID_PARAMETER_VALUE;
          }

          parsedParameters.insert(propertyName);
        }

      finish_material_parsing:;

        if (parsedParameters.size() != materialProperties.size())
        {
          std::string missingProperties;
          for (MaterialPropertiesMap::const_iterator checkName = materialProperties.begin(); checkName != materialProperties.end(); ++checkName)
          { 
            if (parsedParameters.find(checkName->first) == parsedParameters.end())
            {
              missingProperties += missingProperties.empty() ? "" : ", ";
              missingProperties += "'" + checkName->first + "'";
            }
          }

          fprintf(stderr, "Not enough (%d) material parameters for material '%s' at line %d, missing properties are %s\n", materialProperties.size() - parsedParameters.size(), name.c_str(), line, missingProperties.c_str());
          return -OBJ_IMPORT_MATERIAL_NOT_ENOUGH_PARAMETERS;
        }
        
        materials.push_back(m);
        material2IndexMap[name] = materials.size() - 1;

        if (retry_current_line)
          goto reparse_line;
      }
      break;

      default:
        break;
      }
    
    }

    file.close();

    // Preparing scene.
    subject::scene_t* result = subject::scene_create();
    *scene = result;

    // Triangulation to vertices from face indices.
    result->n_faces = faces.size();
    result->faces = (subject::face_t *)malloc(result->n_faces * sizeof(subject::face_t));
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
    result->meshes = (subject::mesh_t *)malloc(sizeof(subject::mesh_t) * result->n_meshes);
    if (meshes.empty())
    {
      result->meshes[0].first_idx = 0;
      result->meshes[0].material_idx = 0;
      result->n_faces = result->meshes[0].n_faces = faces.size();
    }
    else
    {
      meshes.back().n_faces = faces.size() - meshes.back().first_idx;
      memcpy(result->meshes, meshes.data(), sizeof(subject::mesh_t) * result->n_meshes);
    }

    result->n_materials = std::max(1, (int)materials.size());
    result->materials = (subject::material_t*)malloc(sizeof(subject::material_t) * result->n_materials);
    if (materials.empty())
    {
      result->materials[0] = subject::black_body();
    }
    else
    {
      memcpy(result->materials, materials.data(), sizeof(subject::material_t) * result->n_materials);
    }

    return OBJ_IMPORT_OK;
  }
  /// @note Mindflow mode off

  /// @detail Copy-paste from here https://code.google.com/p/ea-utils/source/browse/trunk/clipper/getline.c
  /* Read up to (and including) a TERMINATOR from STREAM into *LINEPTR
  + OFFSET (and null-terminate it). *LINEPTR is a pointer returned from
  malloc (or NULL), pointing to *N characters of space.  It is realloc'd
  as necessary.  Return the number of characters read (not including the
  null terminator), or -1 on error or EOF.  */

  int getstr(char ** lineptr, size_t *n, FILE * stream, char terminator, int offset)
  {
    int nchars_avail;             /* Allocated but unused chars in *LINEPTR.  */
    char *read_pos;               /* Where we're reading into *LINEPTR. */
    int ret;

    if (!lineptr || !n || !stream)
      return -1;

    if (!*lineptr)
    {
      *n = 64;
      *lineptr = (char *)malloc(*n);
      if (!*lineptr)
        return -1;
    }

    nchars_avail = *n - offset;
    read_pos = *lineptr + offset;

    for (;;)
    {
      int c = getc(stream);

      /* We always want at least one char left in the buffer, since we
      always (unless we get an error while reading the first char)
      NUL-terminate the line buffer.  */

      if (nchars_avail < 1)
      {
        if (*n > 64)
          *n *= 2;
        else
          *n += 64;

        nchars_avail = *n + *lineptr - read_pos;
        *lineptr = (char *)realloc(*lineptr, *n);
        if (!*lineptr)
          return -1;
        read_pos = *n - nchars_avail + *lineptr;
      }

      if (c == EOF || ferror(stream))
      {
        /* Return partial line, if any.  */
        if (read_pos == *lineptr)
          return -1;
        else
          break;
      }

      *read_pos++ = c;
      nchars_avail--;

      if (c == terminator)
        /* Return the line.  */
        break;
    }

    /* Done - NUL terminate and return the number of chars read.  */
    *read_pos = '\0';

    ret = read_pos - (*lineptr + offset);
    return ret;
  }

  size_t getline(char **lineptr, size_t *n, FILE *stream)
  {
    return getstr(lineptr, n, stream, '\n', 0);
  }

  int task(FILE* in, thermal_solution::task_t* t)
  {
    char * line = NULL;
    size_t len = 0;
    size_t read;
    size_t n = 0;
    size_t left = 0;
    
    while ((read = getline(&line, &len, in)) != -1)
    {
      switch (line[0])
      {
      case 'n':
        left += n;
        n = 0;
        break;

      case 's':
        if (sscanf(line, "step %f", &t->time_delta) != 1)
        {
          free(line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }
        break;

      case 't':
      { 
        if (left == 0)
        {
          left = 8;
          t->temperatures = (float*)realloc(t->temperatures, (n + left) * sizeof(float));
        }

        if (sscanf(line, "tmprt %f", t->temperatures + (n++)) != 1)
        {
          free(line);
          return -OBJ_IMPORT_FORMAT_ERROR;
        }

        --left;
      }
      break;

      case '\0':
      case '#':
      default:
        break;
      }
    }

    free(line);
    return n;
  }
}
