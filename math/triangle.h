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

#pragma once

#include "types.h"

namespace math
{
  #define TRIANGLE_INTERSECTION_UNIQUE 0
  #define TRIANGLE_INTERSECTION_DISJOINT 1
  #define TRIANGLE_INTERSECTION_DEGENERATE 2
  #define TRIANGLE_INTERSECTION_SAME_PLAIN 3

  /// @brief Find the 3D intersection of a ray with a triangle if any.
  /// @return  -1 = triangle is degenerate (a segment or point)
  ///           0 = disjoint (no intersect)
  ///           1 = intersect in unique point
  ///           2 = ray and triangle are in the same plane
  /// @note Copy-paste from "http://geomalgorithms.com/a06-_intersect-2.html#intersect3D_RayTriangle()"
  int triangle_intersect(ray_t ray, triangle_t triangle, vec3* point);

  /// @brief Calculates triangle center.
  vec3 triangle_center(triangle_t triangle);

  /// @brief Find triangle area.
  float triangle_area(triangle_t triangle);

  /// @brief Creates ray looking from origin point to triangle center with length equal to half of distance.
  ray_t ray_to_triangle(vec3 origin, triangle_t t);

  /// @brief Calculate triangle axes aligned bounding box.
  aabb_t triangle_aabb(const triangle_t& t);

  /// @brief Triangle normal (not normalized).
  vec3 triangle_normal(const triangle_t& t);

  /// @brief Reorder points to flip normal direction.
  void triangle_flip_normal(triangle_t& t);

  /// @brief Set subject normal to the same half-space as sample.
  /// @detail Complexity is two cross and one dot products.
  void triangle_unify_normals(const triangle_t& sample, triangle_t& subject);
}
