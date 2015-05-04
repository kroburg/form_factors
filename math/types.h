#pragma once

namespace math
{
  typedef float point_t;
  struct vec3
  {
    point_t x;
    point_t y;
    point_t z;
  };

  inline vec3 make_vec3(point_t x, point_t y, point_t z)
  {
    return {x, y, z};
  }

  struct triangle_t
  {
    vec3 points[3];
  };

  struct ray_t
  {
    vec3 origin;
    vec3 direction;
  };
}