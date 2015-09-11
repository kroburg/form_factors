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

namespace subject
{
  struct unify_normals_param_t
  {
    face_t* faces;
    int flip_count;
  };

  int unify_normals_walker(int current_idx, int leaf_idx, int mapping, bool have_more, unify_normals_param_t* param)
  {
    if (leaf_idx == -1)
      return 0;

    if (!math::triangle_has_unidirectrional_normals(mapping))
    {
      face_t& leaf_face = param->faces[leaf_idx];
      triangle_flip_normal(leaf_face);
      ++param->flip_count;
    }

    return 0;
  }

  int face_unify_normals(face_t* faces, int n_faces)
  {
    unify_normals_param_t param = { faces, 0 };
    face_walk_graph_n2c(faces, n_faces, (face_graph_walker)unify_normals_walker, &param);
    return param.flip_count;
  }

  int mesh_unify_normals(scene_t* scene, int mesh_idx)
  {
    const mesh_t& mesh = scene->meshes[mesh_idx];
    return face_unify_normals(&scene->faces[mesh.first_idx], mesh.n_faces);
  }
}
