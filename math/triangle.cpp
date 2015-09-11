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
 * This module file contains triangle and ray intersection checks.
 */

#include "triangle.h"
#include "operations.h"
#include <cmath>

namespace math
{
  int triangle_intersect(ray_t ray, triangle_t triangle, vec3* point)
  {
    vec3 u, v, n; // triangle vec3s
    vec3 dir, w0, w; // ray vec3s
    float r, a, b; // params to calc ray-plane intersect

    // get triangle edge vec3s and plane normal
    u = triangle.points[1] - triangle.points[0];
    v = triangle.points[2] - triangle.points[0];
    n = cross(u, v);              // cross product
    if (dot(n, n) < _FLT_EPSILON)      // triangle is degenerate
      return -TRIANGLE_INTERSECTION_DEGENERATE; // do not deal with this case

    dir = ray.direction - ray.origin; // ray direction vec3
    w0 = ray.origin - triangle.points[0];
    a = -dot(n, w0);
    b = dot(n, dir);
    if (fabs(b) < _FLT_EPSILON) { // ray is  parallel to triangle plane
      if (a == 0)            // ray lies in triangle plane
        return -TRIANGLE_INTERSECTION_SAME_PLAIN;
      else return -TRIANGLE_INTERSECTION_DISJOINT; // ray disjoint from plane
    }

    // get intersect point of ray with triangle plane
    r = a / b;
    if (r < 0.0) // ray goes away from triangle
      return -TRIANGLE_INTERSECTION_DISJOINT; // => no intersect
    // for a segment, also test if (r > 1.0) => no intersect

    *point = ray.origin + r * dir; // intersect point of ray and plane

    // is point inside triangle?
    float    uu, uv, vv, wu, wv, D;
    uu = dot(u, u);
    uv = dot(u, v);
    vv = dot(v, v);
    w = *point - triangle.points[0];
    wu = dot(w, u);
    wv = dot(w, v);
    D = uv * uv - uu * vv;

    // get and test parametric coords
    float s, t;
    s = (uv * wv - vv * wu) / D;
    if (s < 0.0 || s > 1.0)         // point is outside T
      return -TRIANGLE_INTERSECTION_DISJOINT;
    t = (uv * wu - uu * wv) / D;
    if (t < 0.0 || (s + t) > 1.0)  // point is outside T
      return -TRIANGLE_INTERSECTION_DISJOINT;

    return TRIANGLE_INTERSECTION_UNIQUE; // point is in T
  }

  vec3 triangle_center(triangle_t t)
  {
    return (t.points[0] + t.points[1] + t.points[2]) / 3.f;
  }

  float triangle_area(triangle_t triangle)
  {
    vec3 v = triangle.points[1] - triangle.points[0];
    vec3 w = triangle.points[2] - triangle.points[0];
    vec3 n = cross(v, w);
    return sqrtf(dot(n, n)) / 2;
  }

  ray_t ray_to_triangle(vec3 origin, triangle_t t)
  {
    vec3 center = triangle_center(t);
    vec3 destination = origin + (center - origin) / 2.f; // ray direction vec3
    return{ origin, destination };
  }

  aabb_t triangle_aabb(const triangle_t& t)
  {
    return{ min(min(t.points[0], t.points[1]), t.points[2]), max(max(t.points[0], t.points[1]), t.points[2]) };
  }

  vec3 triangle_normal(const triangle_t& t)
  {
    vec3 v0 = t.points[1] - t.points[0];
    vec3 v1 = t.points[2] - t.points[0];
    return cross(v0, v1);
  }

  void triangle_flip_normal(triangle_t& t)
  {
    swap(t.points[1], t.points[2]);
  }

  void triangle_unify_normals(const triangle_t& sample, triangle_t& subject)
  {
    vec3 sample_norm = triangle_normal(sample);
    vec3 subject_norm = triangle_normal(subject);
    if (dot(sample_norm, subject_norm) < 0)
      triangle_flip_normal(subject);
  }

  float triangle_least_side_square(const triangle_t& t)
  {
    vec3 v0 = t.points[1] - t.points[0];
    vec3 v1 = t.points[2] - t.points[1];
    vec3 v2 = t.points[0] - t.points[2];

    float d0 = dot(v0, v0);
    float d1 = dot(v1, v1);
    float d2 = dot(v2, v2);

    if (d0 <= d1)
    {
      return d0 < d2 ? d0 : d2;
    }
    else
    {
      return d1 < d2 ? d1 : d2;
    }
  }

  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p)
  {
    float scale = 1.f / triangle_least_side_square(t);
    return triangle_find_adjacent_vertex(t, p, scale);
  }

  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p, float triangle_scale)
  {
    vec3 v0 = t.points[0] - p;
    float d0 = dot(v0, v0);
    if (d0 * triangle_scale < FLT_EPSILON)
      return 0;

    vec3 v1 = t.points[1] - p;
    float d1 = dot(v1, v1);
    if (d1 * triangle_scale < FLT_EPSILON)
      return 1;

    vec3 v2 = t.points[2] - p;
    float d2 = dot(v2, v2);
    if (d2 * triangle_scale < FLT_EPSILON)
      return 2;

    return -1;
  }

  int triangle_find_adjacent_vertices(const triangle_t& l, const triangle_t& r)
  {
    int m = 0;
    float scale = 1.f / triangle_least_side_square(r);

    int a = -1;
    if ((a = triangle_find_adjacent_vertex(r, l.points[0], scale)) >= 0)
      m |= 1 << a;

    if ((a = triangle_find_adjacent_vertex(r, l.points[1], scale)) >= 0)
      m |= 1 << (a + 3);

    if ((a = triangle_find_adjacent_vertex(r, l.points[2], scale)) >= 0)
      m |= 1 << (a + 6);

    return m;
  }

  bool triangle_has_adjacent_edge(int vertex_mapping)
  {
    return (vertex_mapping & (vertex_mapping - 1)) != 0;
  }

#define MAKE_VERTEX_MAPPING2(l1, l2, r1, r2) (1 << (r1 + 3 * l1) | 1 << (r2 + 3 * l2))
#define MAKE_VERTEX_MAPPING3(r1, r2, r3) (1 << r1 | 1 << (r2 + 3) | 1 << (r3 + 6))

  bool triangle_has_unidirectrional_normals(int vertex_mapping)
  {
    switch (vertex_mapping)
    {
    case MAKE_VERTEX_MAPPING2(0, 1, 1, 0):
    case MAKE_VERTEX_MAPPING2(0, 1, 2, 1):
    case MAKE_VERTEX_MAPPING2(0, 1, 0, 2):
    case MAKE_VERTEX_MAPPING2(1, 2, 1, 0):
    case MAKE_VERTEX_MAPPING2(1, 2, 2, 1):
    case MAKE_VERTEX_MAPPING2(1, 2, 0, 2):
    case MAKE_VERTEX_MAPPING2(2, 0, 1, 0):
    case MAKE_VERTEX_MAPPING2(2, 0, 2, 1):
    case MAKE_VERTEX_MAPPING2(2, 0, 0, 2):
    case MAKE_VERTEX_MAPPING3(0, 1, 2):
    case MAKE_VERTEX_MAPPING3(2, 0, 1):
    case MAKE_VERTEX_MAPPING3(1, 2, 0):
      return true;
      break;

      default:
        return false;
    }
  }

  bool triangle_has_adjacent_edge(const triangle_t& l, const triangle_t& r)
  {
    int m = triangle_find_adjacent_vertices(l, r);
    return triangle_has_adjacent_edge(m);
  }
}
