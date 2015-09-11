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

  float build_meshes_areas(const scene_t* scene, float** areas)
  {
    free(*areas);
    *areas = (float*)malloc(sizeof(float) * scene->n_meshes);

    float total_area = 0;
    const int n_meshes = scene->n_meshes;
    for (int m = 0; m != n_meshes; ++m)
    {
      float area = mesh_area(scene, m);
      total_area += area;
      *areas[m] = area;
    }

    return total_area;
  }

  void build_faces_areas(const scene_t* scene, float** areas)
  {
    free(*areas);
    *areas = (float*)malloc(sizeof(float) * scene->n_faces);

    const int n_faces = scene->n_faces;
    for (int f = 0; f != n_faces; ++f)
    {
      float area = math::triangle_area(scene->faces[f]);
      *areas[f] = area;
    }
  }

  void build_face_to_mesh_index(int n_faces, int n_meshes, const mesh_t* meshes, int** index)
  {
    free(*index);
    *index = (int*)malloc(n_faces * sizeof(int));

    // fill face-to-mesh inverted index for every mesh
    for (int m = 0; m != n_meshes; ++m)
    {
      const mesh_t& mesh = meshes[m];
      const int mesh_n_faces = mesh.n_faces;
      for (int f = 0; f != mesh_n_faces; ++f)
      {
        *index[mesh.first_idx + f] = m;
      }
    }
  }

  const material_t& mesh_material(const scene_t* scene, int mesh_idx)
  {
    return scene->materials[scene->meshes[mesh_idx].material_idx];
  }

  
  int face_walk_graph_n2c(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
  { 
    for (int e = 0; e != n_faces; ++e)
    {
      int r = 0;
      int adjacent_face_idx = -1;
      int vertex_mapping = 0;
      const face_t& current = faces[e];
      for (int i = e + 1; i != n_faces; ++i)
      {
        const face_t& test = faces[i];
        int m = triangle_find_adjacent_vertices(current, test);
        if (math::triangle_has_adjacent_edge(m))
        {
          if (adjacent_face_idx != -1)
          {
            if ((r = walker(e, adjacent_face_idx, vertex_mapping, true, param)) != 0)
              return r;
          }
            
          adjacent_face_idx = i;
          vertex_mapping = m;
        }
      }

      // report last adjacent face or -1
      if ((r = walker(e, adjacent_face_idx, vertex_mapping, false, param)) != 0)
        return r;
    }

    return 0;
  }

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
