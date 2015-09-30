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

  /// @brief Square length of least triangle side.
  float triangle_least_side_square(const triangle_t& t);

  /**
    @brief Find triangle vertex closest to point.
    @return -1 if point is not close enought, vertex index (0-2) otherwise.
  */
  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p);

  /**
    @brief Find triangle vertex closest to point.
    @param triangle_scale 1/triangle specific size.
    @return -1 if point is not close enought, vertex index (0-2) otherwise.
  */
  int triangle_find_adjacent_vertex(const triangle_t& t, const vec3& p, float triangle_scale);

  /**
  @brief Make vertex mapping for adjacent vertices.
  @param p1 adjacent face vertex index which is mapped to first face vertex.
  */
  int make_vertex_mapping_123(char p1, char p2, char p3);
  int make_vertex_mapping_13(char p1, char p3);
  int make_vertex_mapping_12(char p1, char p2);
  int make_vertex_mapping_23(char p2, char p3);

  /**
    @brief Find adjacent vertices of two triangles.
    @return Bitmap of adjacent vertices (l[2]?r[2], l[2]?r[1], l[2]?r[0], l[1]?r[2], l[1]?r[1], l[1]?r[0], l[0]?r[2], l[0]?r[1], l[0]?r[0]).
  */
  int triangle_find_adjacent_vertices(const triangle_t& l, const triangle_t& r);

  /**
    @brief Check two triangles vertex mapping for adjacent edge existence.
    @param vertex_mapping mapping from triangle_find_adjacent_vertices() call.
  */
  bool triangle_has_adjacent_edge(int vertex_mapping);

  /**
    @brief Check two triangles has unidirectional normals.
    @param vertex_mapping mapping from triangle_find_adjacent_vertices() call.
  */
  bool triangle_has_unidirectrional_normals(int vertex_mapping);

  /// @brief Check two triangles for adjacent edge existence.
  bool triangle_has_adjacent_edge(const triangle_t& l, const triangle_t& r);
}
