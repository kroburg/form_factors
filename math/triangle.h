# Copyright 2015 Stepan Tezyunichev (stepan.tezyunichev@gmail.com).
# This file is part of form_factors.
#
# form_factors is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# form_factors is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with form_factors.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include "types.h"

namespace math
{
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
