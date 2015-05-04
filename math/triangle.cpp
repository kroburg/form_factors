#include "triangle.h"
#include "operations.h"
#include <cmath>

#define EPSILON   0.00000001

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
    if (dot(n, n) < EPSILON)      // triangle is degenerate
      return -TRIANGLE_INTERSECTION_DEGENERATE; // do not deal with this case

    dir = ray.direction - ray.origin; // ray direction vec3
    w0 = ray.origin - triangle.points[0];
    a = -dot(n, w0);
    b = dot(n, dir);
    if (fabs(b) < EPSILON) { // ray is  parallel to triangle plane
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

  ray_t ray_to_triangle(vec3 origin, triangle_t t)
  {
    vec3 center = triangle_center(t);
    vec3 destination = origin + (center - origin) / 2.f; // ray direction vec3
    return{ origin, destination };
  }
}
