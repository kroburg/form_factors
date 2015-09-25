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
#include "../math/operations.h"
#include "../math/triangle.h"
#include <cmath>

namespace subject
{
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

  struct buffered_face_point
  {
    math::vec3 point;

    bool operator < (const buffered_face_point& r) const
    {
      const math::vec3 size = max(abs(point), abs(r.point));
      math::vec3 diff = (point - r.point) / size;
      if (diff.x < -FLT_EPSILON)
        return true;
      else if (diff.x > FLT_EPSILON)
        return false;
      else if (diff.y < -FLT_EPSILON)
        return true;
      else if (diff.y > FLT_EPSILON)
        return false;
      else if (diff.z < -FLT_EPSILON)
        return true;
      else
        return false;
    }
  };


  int face_walk_graph_nlgn(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
  {
    return 0;
  }
}