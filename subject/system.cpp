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
* This module contains basic types for scene definition and manipulation.
*/

#include "system.h"
#include "../math/triangle.h"
#include <stdlib.h>

namespace subject
{
  shell_properties_t default_shell_properties()
  {
    return { 1.f, 1.f, 1.f, 1.f };
  }

  optical_properties_t black_material()
  {
    return{ 0.f, 0.f, 1.f, 0.f, 1.f };
  }

  material_t black_body()
  {
    return { default_shell_properties(), black_material(), black_material() };
  }

  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)malloc(sizeof(scene_t));
    *s = { 0, 0 // faces
      , 0, 0 // materials
      , 0, 0 // meshes
    };
    return s;
  }

  void scene_free(scene_t* scene)
  {
    if (scene)
    {
      free(scene->faces);
      free(scene->meshes);
      free(scene->materials);
      free(scene);
    }
  }

  float mesh_area(const scene_t* scene, const mesh_t& mesh)
  {
    float area = 0;
    for (int f = 0; f != mesh.n_faces; ++f)
    {
      area += math::triangle_area(scene->faces[mesh.first_idx + f]);
    }
    return area;
  }

  float mesh_area(const scene_t* scene, int mesh_idx)
  {
    return mesh_area(scene, scene->meshes[mesh_idx]);
  }

  const material_t& mesh_material(const scene_t* scene, int mesh_idx)
  {
    return scene->materials[scene->meshes[mesh_idx].material_idx];
  }
}
