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
 * This module contains math operations on vectors.
 */

#pragma once

#include "types.h"
#include "operations.h"
#include <cmath>
#include <algorithm>

/**
 * Precision of floating point variable.
 */
#define _FLT_EPSILON   0.00000001

namespace math
{
  /// @brief Dot product of two 3d vectors.
  inline float dot(vec3 a, vec3 b)
  {
    return a.x * b.x + a.y * b.y + a.z * b.z;
  }

  /// @brief Cross product of two 3d vectors.
  inline vec3 cross(vec3 a, vec3 b)
  {
    return make_vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
  }

  /// @brief 3d vector norm.
  inline point_t norm(vec3 a)
  {
    return sqrtf(dot(a, a));
  }

  /// @brief Sum of two 3d vectors.
  inline vec3 operator+(vec3 a, vec3 b)
  {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
  }

  /// @brief Sum-and-assign operation of two 3d vectors.
  inline void operator+=(vec3&a, vec3 b)
  {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
  }

  /// @brief Minus operation of two 3d vectors.
  inline vec3 operator-(vec3 a, vec3 b)
  {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
  }

  /// @brief Minus-and-assign operation of two 3d vectors.
  inline void operator-=(vec3&a, vec3 b)
  {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
  }

  /// @brief 3d vector to scalar multiplication.
  inline vec3 operator*(vec3 a, point_t b)
  {
    return make_vec3(a.x * b, a.y * b, a.z * b);
  }

  /// @brief 3d vector to scalar multiplication.
  inline vec3 operator*(point_t b, vec3 a)
  {
    return a * b;
  }

  /// @brief 3d vector to scalar division.
  inline vec3 operator/(vec3 a, point_t b)
  {
    return make_vec3(a.x / b, a.y / b, a.z / b);
  }

  /// @brief Division-and-assign operation of 3d vector and scalar.
  inline void operator/=(vec3& a, point_t b)
  {
    b = point_t(1) / b;
    a.x *= b;
    a.y *= b;
    a.z *= b;
  }

  /// @brief 3d vector containing minimum coordinates of two 3d vectors.
  inline vec3 min(vec3 a, vec3 b)
  {
    return make_vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
  }

  /// @brief 3d vector containing maximum coordinates of two 3d vectors.
  inline vec3 max(vec3 a, vec3 b)
  {
    return make_vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
  }

  /// @brief Normalizes 3d vector.
  inline vec3 normalize(vec3 a)
  {
    point_t n = norm(a);
    if (n > _FLT_EPSILON) {
      a /= sqrtf(n);
    }
    return a;
  }

  /// @brief Checks for vector equality.
  inline bool near_enough(vec3 a, vec3 b)
  {
    vec3 dist = a - b;
    return dot(dist, dist) < _FLT_EPSILON;
  }

  /// @brief Union of two aabb (aabb to containing both aabb).
  inline aabb_t operator+(const aabb_t& l, const aabb_t& r)
  {
    return { min(l.min, r.min), max(l.max, r.max) };
  }
}
