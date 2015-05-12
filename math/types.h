// Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
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

#pragma once

#define M_PI   3.14159265358979323846264338327950288
#define M_2PI  6.28318530717958647692528676655900576

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

  struct mat33
  {
    point_t p[3][3];
  };

  inline mat33 make_mat33(point_t a00, point_t a01, point_t a02, point_t a10, point_t a11, point_t a12, point_t a20, point_t a21, point_t a22)
  {
    return{ { { a00, a01, a02 }, { a10, a11, a12 }, { a20, a21, a22 } } };
  }
}
