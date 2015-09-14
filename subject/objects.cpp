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
}