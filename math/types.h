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
 * This module contains some basic types - vectors, points and constructing operations
 */

#pragma once

///
#ifdef _WIN32
#define M_PI   3.14159265358979323846264338327950288
#define M_2PI  6.28318530717958647692528676655900576
#endif // _WIN32

namespace math
{
  /// @brief Scalar type definition (single or double precision)
  typedef float point_t;

  /// @brief 3d vector type
  struct vec3
  {
    point_t x;
    point_t y;
    point_t z;
  };

  /// @brief 3d vector constructor from 3 scalar points
  inline vec3 make_vec3(point_t x, point_t y, point_t z)
  {
    vec3 result = {x, y, z};
    return result;
  }

  /// @brief Triangle type
  struct triangle_t
  {
    vec3 points[3];
  };

  /// @brief Ray type
  struct ray_t
  {
    vec3 origin; ///< Origin point of ray
    vec3 direction; ///< Ray direction
  };

  /// @brief 3x3 matrix type
  struct mat33
  {
    point_t p[3][3]; ///< 3 rows of 3 columns each
  };

  /// @brief Creates 3x3 matrix from 3 scalar points
  inline mat33 make_mat33(point_t a00, point_t a01, point_t a02, point_t a10, point_t a11, point_t a12, point_t a20, point_t a21, point_t a22)
  {
    mat33 result = { { { a00, a01, a02 }, { a10, a11, a12 }, { a20, a21, a22 } } };
    return result;
  }
}
