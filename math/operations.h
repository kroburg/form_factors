#pragma once

#include "types.h"

namespace math
{
  inline float dot(vec3 a, vec3 b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  inline vec3 cross(vec3 a, vec3 b)
  {
    return make_vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
  }

  inline vec3 operator+(vec3 a, vec3 b)
  {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  inline void operator+=(vec3&a, vec3 b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
  }

  inline vec3 operator-(vec3 a, vec3 b)
  {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  inline void operator-=(vec3&a, vec3 b)
  {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
  }

  inline vec3 operator*(vec3 a, point_t b)
  {
    return make_vec3(a.x * b, a.y * b, a.z * b);
  }

  inline vec3 operator*(point_t b, vec3 a)
  {
    return a * b;
  }

  inline vec3 operator/(vec3 a, point_t b)
  {
    return make_vec3(a.x / b, a.y / b, a.z / b);
  }

  inline bool near_enough(vec3 a, vec3 b)
  {
    vec3 dist = a - b;
    return dot(dist, dist) < 0.000001f;
  }

#define TRIANGLE_INTERSECTION_UNIQUE 0
#define TRIANGLE_INTERSECTION_DISJOINT 1
#define TRIANGLE_INTERSECTION_DEGENERATE 2
#define TRIANGLE_INTERSECTION_SAME_PLAIN 3

  // @todo Fix copy-pasted documentation.
  // @note Copy-paste from http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()
  // intersect_triangle(): find the 3D intersection of a ray with a triangle
  //    Return: -1 = triangle is degenerate (a segment or point)
  //             0 =  disjoint (no intersect)
  //             1 =  intersect in unique point
  //             2 =  are in the same plane
  int triangle_intersect(ray_t ray, triangle_t triangle, vec3* point);
  
  /// @brief Calculate triangle center.
  vec3 triangle_center(triangle_t triangle);

  /// @brief Make ray looking from origin point to triangle center with length equal to half of distance.
  ray_t ray_to_triangle(vec3 origin, triangle_t t);
}
