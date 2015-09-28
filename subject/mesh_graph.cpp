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
#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <vector>
#include <assert.h>

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

  struct indexed_walker
  {
    struct face_point
    {
      int point;
      char index;
      bool operator<(const face_point& r) const
      {
        return point < r.point;
      }
    };

    struct indexed_face_t
    {
      int points[3];
    };

    typedef std::map<math::vec3, int> point_index;
    typedef std::set<face_point> united_faces;
    typedef std::map<int, united_faces> point_face_index;

    point_index points;
    std::vector<indexed_face_t> indexed_faces;
    point_face_index p2f;

    int index_point(math::vec3 p)
    {
      typedef std::pair<point_index::iterator, bool> point_insertion;
      point_insertion r = points.insert(point_index::value_type(p, 0));
      if (r.second)
        r.first->second = (int)points.size();
      return r.first->second;
    }

    int face_edge(int f_idx, int p1, int p2)
    {
      const indexed_face_t&f = indexed_faces[f_idx];
      if (f.points[0] == p1 || f.points[0] == p2)
      {
        if (f.points[1] == p1 || f.points[1] == p2)
        {
          assert(!(f.points[2] == p1 || f.points[2] == p2));
          return 1 << 0;
        }
        else if (f.points[2] == p1 || f.points[2] == p2)
        {
          assert(!(f.points[1] == p1 || f.points[1] == p2));
          return 1 << 2;
        }
        else
        {
          assert(!"Invalid face points");
          return -1;
        }
      }
      else
      {
        assert(f.points[1] == p1 || f.points[1] == p2);
        assert(f.points[2] == p1 || f.points[2] == p2);
        return 1 << 1;
      }
    }

    void collect_adjacents(int source, int p1, int p2)
    {
      const united_faces& f1 = p2f[p1];
      const united_faces& f2 = p2f[p2];
      std::vector<face_point> common;
      std::set_intersection(f1.begin(), f1.end(), f2.begin(), f2.end(), std::back_inserter(common));
      for (face_point f : common)
      {
        if (f.point != source);

      }
    }

    int face_walk_graph_nlgn(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
    {
      build_face_points_index(faces, n_faces);

      for (int f = 0; f != n_faces; ++f)
      {


      }

      return 0;
    }

    void build_face_points_index(const face_t* faces, int n_faces)
    {
      indexed_faces.resize(n_faces);

      for (int f = 0; f != n_faces; ++f)
      {
        for (char p = 0; p != 3; ++p)
        {
          int point_idx = index_point(faces[f].points[p]);
          indexed_faces[f].points[p] = point_idx;
          face_point entry = { f, p };
          p2f[point_idx].insert(entry);
        }
      }
    }
  };
}