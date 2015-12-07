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

  float triangle_area_xy(triangle_t triangle)
  {
    triangle.points[0].z = 0;
    triangle.points[1].z = 0;
    triangle.points[2].z = 0;
    return triangle_area(triangle);
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

  int make_vertex_mapping_123(char p1, char p2, char p3)
  {
    return 1 << p1 | 1 << (p2 + 3) | 1 << (p3 + 6);
  }

  int make_vertex_mapping_13(char p1, char p3)
  {
    return 1 << p1 | 1 << (p3 + 6);
  }

  int make_vertex_mapping_12(char p1, char p2)
  {
    return 1 << p1 | 1 << (p2 + 3);
  }

  int make_vertex_mapping_23(char p2, char p3)
  {
    return 1 << (p2 + 3) | 1 << (p3 + 6);
  }

  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p)
  {
    float scale = 1.f / triangle_least_side_square(t);
    return triangle_find_adjacent_vertex(t, p, scale);
  }

  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p, float triangle_scale)
  {
    // @todo Incorrect algorithm. It must use max(abs()) of coordinate.
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

#define DECODE_VERTEX_MAPPING2(l1, l2, r1, r2) lp[0] = l1, lp[1] = l2, rp[0] = r1, rp[1] = r2; return 2; break;
#define DECODE_VERTEX_MAPPING3(r1, r2, r3) lp[0] = 0, lp[1] = 1, lp[2] = 2, rp[0] = r1, rp[1] = r2, rp[2] = r3; return 3; break;

  int decode_vertex_mapping(int vertex_mapping, char*lp, char* rp)
  {
    switch (vertex_mapping)
    {
    case MAKE_VERTEX_MAPPING2(0, 1, 0, 1):
      DECODE_VERTEX_MAPPING2(0, 1, 0, 1);
    case MAKE_VERTEX_MAPPING2(0, 1, 1, 0):
      DECODE_VERTEX_MAPPING2(0, 1, 1, 0);

    case MAKE_VERTEX_MAPPING2(0, 1, 1, 2):
      DECODE_VERTEX_MAPPING2(0, 1, 1, 2);
    case MAKE_VERTEX_MAPPING2(0, 1, 2, 1):
      DECODE_VERTEX_MAPPING2(0, 1, 2, 1);

    case MAKE_VERTEX_MAPPING2(0, 1, 0, 2):
      DECODE_VERTEX_MAPPING2(0, 1, 0, 2);
    case MAKE_VERTEX_MAPPING2(0, 1, 2, 0):
      DECODE_VERTEX_MAPPING2(0, 1, 2, 0);

    case MAKE_VERTEX_MAPPING2(1, 2, 0, 1):
      DECODE_VERTEX_MAPPING2(1, 2, 0, 1);
    case MAKE_VERTEX_MAPPING2(1, 2, 1, 0):
      DECODE_VERTEX_MAPPING2(1, 2, 1, 0);

    case MAKE_VERTEX_MAPPING2(1, 2, 1, 2):
      DECODE_VERTEX_MAPPING2(1, 2, 1, 2);
    case MAKE_VERTEX_MAPPING2(1, 2, 2, 1):
      DECODE_VERTEX_MAPPING2(1, 2, 2, 1);

    case MAKE_VERTEX_MAPPING2(1, 2, 0, 2):
      DECODE_VERTEX_MAPPING2(1, 2, 0, 2);
    case MAKE_VERTEX_MAPPING2(1, 2, 2, 0):
      DECODE_VERTEX_MAPPING2(1, 2, 2, 0);

    case MAKE_VERTEX_MAPPING2(0, 2, 0, 1):
      DECODE_VERTEX_MAPPING2(0, 2, 0, 1);
    case MAKE_VERTEX_MAPPING2(0, 2, 1, 0):
      DECODE_VERTEX_MAPPING2(0, 2, 1, 0);

    case MAKE_VERTEX_MAPPING2(0, 2, 1, 2):
      DECODE_VERTEX_MAPPING2(0, 2, 1, 2);
    case MAKE_VERTEX_MAPPING2(0, 2, 2, 1):
      DECODE_VERTEX_MAPPING2(0, 2, 2, 1);

    case MAKE_VERTEX_MAPPING2(0, 2, 0, 2):
      DECODE_VERTEX_MAPPING2(0, 2, 0, 2);
    case MAKE_VERTEX_MAPPING2(0, 2, 2, 0):
      DECODE_VERTEX_MAPPING2(0, 2, 2, 0);

    case MAKE_VERTEX_MAPPING3(0, 1, 2):
      DECODE_VERTEX_MAPPING3(0, 1, 2);
    case MAKE_VERTEX_MAPPING3(0, 2, 1):
      DECODE_VERTEX_MAPPING3(0, 2, 1);

    case MAKE_VERTEX_MAPPING3(1, 0, 2):
      DECODE_VERTEX_MAPPING3(1, 0, 2);
    case MAKE_VERTEX_MAPPING3(1, 2, 0):
      DECODE_VERTEX_MAPPING3(1, 2, 0);

    case MAKE_VERTEX_MAPPING3(2, 0, 1):
      DECODE_VERTEX_MAPPING3(2, 0, 1);
    case MAKE_VERTEX_MAPPING3(2, 1, 0):
      DECODE_VERTEX_MAPPING3(2, 1, 0);

    // 0, 1 or invalid format
    default:
      return 0;
    }
  }

  // @todo Not tested.
  int decode_vertex_mapping_cyclic(int vertex_mapping, char*lp, char* rp)
  {
    int mask = 1;
    int count = 0;
    for (int l = 0; l != 3; ++l)
    {
      for (int r = 0; r != 3 && count != 3; ++r, mask << 1)
      {
        if ((vertex_mapping & mask) == 0)
          continue;

        *lp++ = l;
        *rp++ = r;
        ++count;
        mask = mask << (3 - r);
        break;
      }
    }

    return count;
  }

  bool triangle_has_adjacent_edge(const triangle_t& l, const triangle_t& r)
  {
    int m = triangle_find_adjacent_vertices(l, r);
    return triangle_has_adjacent_edge(m);
  }

  aabb_t triangles_aabb(const triangle_t* triangles, int n_triangles)
  {
    aabb_t result = n_triangles ? triangle_aabb(triangles[0]) : aabb_t();
    for (int i = 1; i != n_triangles; ++i)
      result += triangle_aabb(triangles[i]);
    return result;
  }

  sphere_t triangles_bsphere(const triangle_t* triangles, int n_triangles)
  {
    vec3 geom_center = make_vec3(0, 0, 0);

    for (int i = 0; i != n_triangles; ++i)
      geom_center += triangle_center(triangles[i]);
    geom_center /= (float)n_triangles;

    float r = 0;
    for (int i = 1; i != n_triangles; ++i)
    {
      float t = norm(geom_center - triangles[i].points[0]);
      if (t > r)
        r = t;
      t = norm(geom_center - triangles[i].points[1]);
      if (t > r)
        r = t;
      t = norm(geom_center - triangles[i].points[2]);
      if (t > r)
        r = t;
    }

    return{ geom_center, r };
  }

  bool triangle_2d_contains_point(const triangle_t& t, const vec3& p)
  {
    // @todo Not tested!
    const vec3 p1 = t.points[0];
    const vec3 p2 = t.points[1];
    const vec3 p3 = t.points[2];

    float c = ((p2.y - p3.y)*(p1.x - p3.x) + (p3.x - p2.x)*(p1.y - p3.y));
    float alpha = ((p2.y - p3.y)*(p.x - p3.x) + (p3.x - p2.x)*(p.y - p3.y)) / c;
    float beta = ((p3.y - p1.y)*(p.x - p3.x) + (p1.x - p3.x)*(p.y - p3.y)) / c;
    float gamma = 1.0f - alpha - beta;

    return alpha >= 0.f && beta >= 0.f && gamma >= 0.f;
  }

  triangle_t triangle_order(triangle_t t)
  {
    if (t.points[1].y > t.points[0].y)
      swap(t.points[1], t.points[0]);
    if (t.points[2].y > t.points[0].y)
      swap(t.points[2], t.points[0]);
    if (t.points[2].y > t.points[1].y)
      swap(t.points[2], t.points[1]);
    return t;
  }

  triangles_analysis_t triangles_analyze(const triangle_t* triangles, int n_triangles)
  {
    aabb_t aabb = n_triangles ? triangle_aabb(triangles[0]) : aabb_t();
    float max_area = n_triangles ? triangle_area(triangles[0]) : 0;
    float max_area_xy = n_triangles ? triangle_area_xy(triangles[0]) : 0;
    float min_area = max_area;
    float min_area_xy = max_area_xy;
    float average_area = max_area;
    float average_area_xy = max_area_xy;
    triangle_t average = n_triangles ? triangles[0] : triangle_t();
    for (int i = 1; i != n_triangles; ++i)
    {
      aabb += triangle_aabb(triangles[i]);
      float area = triangle_area(triangles[i]);
      float area_xy = triangle_area_xy(triangles[i]);
      min_area = std::min(min_area, area);
      max_area = std::max(max_area, area);
      average_area += area;

      min_area_xy = std::min(min_area_xy, area_xy);
      max_area_xy = std::max(max_area, area_xy);
      average_area_xy += area_xy;

      average.points[0] += triangles[i].points[0];
      average.points[1] += triangles[i].points[1];
      average.points[2] += triangles[i].points[2];
    }

    aabb.max *= 1.0001f;

    average_area /= (float)n_triangles;
    average_area_xy /= (float)n_triangles;

    average.points[0] /= (float)n_triangles;
    average.points[1] /= (float)n_triangles;
    average.points[2] /= (float)n_triangles;

    return{ aabb, min_area, max_area, average_area, min_area_xy, max_area_xy, average_area_xy, average };
  }
}
