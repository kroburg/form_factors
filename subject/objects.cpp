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

#include "objects.h"
#include "system.h"
#include <cstdlib>

namespace subject
{
  math::face_t* unify_object_faces(math::face_t* faces, int n_faces)
  {
    face_unify_normals(faces, n_faces);
    return faces;
  }

  const math::vec3 box_vertices[8] = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } };

  math::face_t raw_box_faces[12] = {
    make_face(box_vertices[0], box_vertices[2], box_vertices[1]), // first face normal point downward
    make_face(box_vertices[0], box_vertices[3], box_vertices[2]),
    make_face(box_vertices[3], box_vertices[2], box_vertices[6]),
    make_face(box_vertices[3], box_vertices[7], box_vertices[6]),
    make_face(box_vertices[0], box_vertices[3], box_vertices[7]),
    make_face(box_vertices[0], box_vertices[4], box_vertices[7]),
    make_face(box_vertices[1], box_vertices[2], box_vertices[6]),
    make_face(box_vertices[1], box_vertices[5], box_vertices[6]),
    make_face(box_vertices[0], box_vertices[1], box_vertices[5]),
    make_face(box_vertices[0], box_vertices[4], box_vertices[5]),
    make_face(box_vertices[4], box_vertices[5], box_vertices[6]),
    make_face(box_vertices[4], box_vertices[7], box_vertices[6])
  };

  const math::face_t* box()
  {
    static math::face_t* faces = unify_object_faces(raw_box_faces, 12);
    return faces;
  }

  math::face_t* plane_grid_faces(float width, float height, int cells_x, int cells_y)
  {
    const int face_count = cells_x * cells_y * 2;
    face_t* faces = (face_t*)malloc(sizeof(face_t) * face_count);

    float step_x = width / cells_x;
    float step_y = height / cells_y;
    for (int i = 0; i != cells_x; ++i)
    {
      for (int j = 0; j != cells_y; ++j)
      {
        float x = step_x * i;
        float y = step_y * j;

        math::vec3 A = math::make_vec3(x, y, 0);
        math::vec3 B = math::make_vec3(x + step_x, y, 0);
        math::vec3 C = math::make_vec3(x + step_x, y + step_y, 0);
        math::vec3 D = math::make_vec3(x, y + step_y, 0);

        faces[(i * cells_x + j) * 2] = make_face(A, B, D);
        faces[(i * cells_x + j) * 2 + 1] = make_face(C, D, B);
      }
    }

    return faces;
  }

  mesh_t* plane_grid_meshes(int cells_x, int cells_y)
  {
    const int mesh_count = cells_x * cells_y;
    mesh_t* meshes = (mesh_t*)malloc(sizeof(mesh_t) * mesh_count);

    for (int i = 0; i != cells_x; ++i)
    {
      for (int j = 0; j != cells_y; ++j)
      {
        int idx = i * cells_x + j;
        meshes[idx] = make_mesh(idx * 2, 2, 0);
      }
    }

    return meshes;
  }

  shell_properties_t default_shell_properties()
  {
    return{ 1.f, 1.f, 1.f, 1.f };
  }

  optical_properties_t black_material()
  {
    return{ 0.f, 0.f, 1.f, 0.f, 1.f };
  }

  shell_properties_t shell_Al()
  {
    shell_properties_t result = default_shell_properties();
    result.density = 2.6989e3f;
    result.heat_capacity = 903;
    result.thermal_conductivity = 237;
    return result;
  }

  optical_properties_t optical_Al()
  {
    optical_properties_t result = black_material();
    result.emissivity = 0.09f;
    
    return result;
  }

  material_t black_body()
  {
    return{ default_shell_properties(), black_material(), black_material(), "black_body" };
  }

  material_t material_Al(float thickness)
  {
    material_t result = { shell_Al(), optical_Al(), optical_Al(), "alliminuim" };
    
    result.shell.thickness = thickness;

    return result;
  }
}