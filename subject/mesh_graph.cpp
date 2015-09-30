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
#include "../math/three_way_compare.h"
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

  class indexed_walker_t
  {
  public:
    indexed_walker_t(const face_t* faces, int n_faces)
    {
      build_index(faces, n_faces);
    }

    int walk(face_graph_walker walker, void* param)
    {
      for (int f = 0; f != (int)indexed_faces.size(); ++f)
      {
        if (int r = collect_adjacents(f, walker, param))
          return r;
      }

      return 0;
    }

  private:
    void build_index(const face_t* faces, int n_faces)
    {
      indexed_faces.resize(n_faces);

      for (int f = 0; f != n_faces; ++f)
      {
        for (char p = 0; p != 3; ++p)
        {
          int point_idx = index_point(faces[f].points[p]);
          indexed_faces[f].points[p] = point_idx;
          face_point_t entry = { f, p };
          p2f[point_idx].insert(entry);
        }
      }
    }

    int index_point(math::vec3 p)
    {
      typedef std::pair<point_index::iterator, bool> point_insertion;
      point_insertion r = points.insert(point_index::value_type(p, 0));
      if (r.second)
        r.first->second = (int)points.size();
      return r.first->second;
    }

    int collect_adjacents(int f0, face_graph_walker walker, void* param)
    {
      const indexed_face_t& face = indexed_faces[f0];
      int p1 = face.points[0];
      int p2 = face.points[1];
      int p3 = face.points[2];

      const united_faces_t& f1 = p2f[p1];
      const united_faces_t& f2 = p2f[p2];
      const united_faces_t& f3 = p2f[p3];

      united_faces_t::const_iterator fi1 = f1.begin();
      united_faces_t::const_iterator fe1 = f1.end();
      united_faces_t::const_iterator fi2 = f2.begin();
      united_faces_t::const_iterator fe2 = f2.end();
      united_faces_t::const_iterator fi3 = f3.begin();
      united_faces_t::const_iterator fe3 = f3.end();

      int adjacent_face = -1;
      int mapping = 0;

      while (fi1 != fe1 && fi2 != fe2 && fi3 != fe3)
      {
        switch (math::three_way_less_t c = math::three_way_less(fi1->face, fi2->face, fi3->face))
        {
        case math::l3w_equal:
          if (f0 < fi1->face)
          {
            if (adjacent_face != -1)
            {
              if (int r = walker(f0, adjacent_face, mapping, true, param))
                return r;
            }

            adjacent_face = fi1->face;
            mapping = math::make_vertex_mapping_123(fi1->point_index, fi2->point_index, fi3->point_index);
          }

          ++fi1;
          ++fi2;
          ++fi3;
          break;
        case math::l3w_first_less:
          ++fi1;
          break;
        case math::l3w_second_less:
          ++fi2;
          break;
        case math::l3w_third_less:
          ++fi3;
          break;
        case math::l3w_first_greater:
          if (f0 < fi2->face)
          {
            if (adjacent_face != -1)
            {
              if (int r = walker(f0, adjacent_face, mapping, true, param))
                return r;
            }

            adjacent_face = fi2->face;
            mapping = math::make_vertex_mapping_23(fi2->point_index, fi3->point_index);
          }

          ++fi2;
          ++fi3;
          break;
        case math::l3w_second_greater:
          if (f0 < fi1->face)
          {
            if (adjacent_face != -1)
            {
              if (int r = walker(f0, adjacent_face, mapping, true, param))
                return r;
            }

            adjacent_face = fi1->face;
            mapping = math::make_vertex_mapping_13(fi1->point_index, fi3->point_index);
          }

          ++fi1;
          ++fi3;
          break;
        case math::l3w_third_greater:
          if (f0 < fi1->face)
          {
            if (adjacent_face != -1)
            {
              if (int r = walker(f0, adjacent_face, mapping, true, param))
                return r;
            }

            adjacent_face = fi1->face;
            mapping = math::make_vertex_mapping_12(fi1->point_index, fi2->point_index);
          }

          ++fi1;
          ++fi2;
          break;
        }
      }

      int(*make_vertex_mapping)(char, char) = &math::make_vertex_mapping_12;
      if (fi1 == fe1)
      {
        fi1 = fi2;
        fe1 = fe2;
        fi2 = fi3;
        fe2 = fe3;
        make_vertex_mapping = &math::make_vertex_mapping_23;
      }
      else if (fi2 == fe2)
      {
        fi2 = fi3;
        fe2 = fe3;
        make_vertex_mapping = &math::make_vertex_mapping_13;
      }

      while (fi1 != fe1 && fi2 != fe2)
      {
        switch (math::two_way_less_t c = math::two_way_less(fi1->face, fi2->face))
        {
        case math::l2w_first_less:
          ++fi1;
          break;

        case math::l2w_second_less:
          ++fi2;
          break;

        case math::l2w_equal:
          if (f0 < fi1->face)
          {
            if (adjacent_face != -1)
            {
              if (int r = walker(f0, adjacent_face, mapping, true, param))
                return r;
            }

            adjacent_face = fi1->face;
            mapping = make_vertex_mapping(fi1->point_index, fi2->point_index);
          }

          ++fi1;
          ++fi2;
          break;
        }
      }


      if (int r = walker(f0, adjacent_face, mapping, false, param))
        return r;

      return 0;
    }

  private:
    struct face_point_t
    {
      int face;
      char point_index;

      bool operator<(const face_point_t& r) const
      {
        return face < r.face;
      }
    };

    struct indexed_face_t
    {
      int points[3];
    };

    typedef std::map<math::vec3, int> point_index;
    typedef std::set<face_point_t> united_faces_t;
    typedef std::map<int, united_faces_t> point_face_index;

    point_index points;
    std::vector<indexed_face_t> indexed_faces;
    point_face_index p2f;
  };

  int face_walk_graph_indexed(const face_t* faces, int n_faces, face_graph_walker walker, void* param)
  {
    indexed_walker_t indexed_walker(faces, n_faces);
    return indexed_walker.walk(walker, param);
  }
}