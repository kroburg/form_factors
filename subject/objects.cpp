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
  const math::face_t* UnifyObjectFaces(math::face_t* faces, int n_faces)
  {
    face_unify_normals(faces, n_faces);
    return faces;
  }

  const math::vec3 BoxVertices[8] = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } };

  math::face_t RawBoxFaces[12] = {
    make_face(BoxVertices[0], BoxVertices[2], BoxVertices[1]), // first face normal point downward
    make_face(BoxVertices[0], BoxVertices[3], BoxVertices[2]),
    make_face(BoxVertices[3], BoxVertices[2], BoxVertices[6]),
    make_face(BoxVertices[3], BoxVertices[7], BoxVertices[6]),
    make_face(BoxVertices[0], BoxVertices[3], BoxVertices[7]),
    make_face(BoxVertices[0], BoxVertices[4], BoxVertices[7]),
    make_face(BoxVertices[1], BoxVertices[2], BoxVertices[6]),
    make_face(BoxVertices[1], BoxVertices[5], BoxVertices[6]),
    make_face(BoxVertices[0], BoxVertices[1], BoxVertices[5]),
    make_face(BoxVertices[0], BoxVertices[4], BoxVertices[5]),
    make_face(BoxVertices[4], BoxVertices[5], BoxVertices[6]),
    make_face(BoxVertices[4], BoxVertices[7], BoxVertices[6])
  };

  const math::face_t* BoxFaces()
  {
    static const math::face_t* faces = UnifyObjectFaces(RawBoxFaces, 12);
    return faces;
  }
}