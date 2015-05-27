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
 * This module contains math operations between matrices and vectors
 */

#pragma once

#include "types.h"
#include "operations.h"
#include <float.h>

namespace math
{
  /// @brief 3x3 matrix to vector multiplication
  inline vec3 operator * (mat33 mat, vec3 vec)
  {
    return make_vec3(
      (mat.p[0][0] * vec.x) + (mat.p[1][0] * vec.y) + (mat.p[2][0] * vec.z),
      (mat.p[0][1] * vec.x) + (mat.p[1][1] * vec.y) + (mat.p[2][1] * vec.z),
      (mat.p[0][2] * vec.x) + (mat.p[1][2] * vec.y) + (mat.p[2][2] * vec.z)
      );
  }

  /// @brief 3x3 matrix to matrix multiplication
  inline mat33 operator * (mat33 a, mat33 b)
  {
    return make_mat33(
      a.p[0][0] * b.p[0][0] + a.p[0][1] * b.p[1][0] + a.p[0][2] * b.p[2][0],
      a.p[1][0] * b.p[0][0] + a.p[1][1] * b.p[1][0] + a.p[1][2] * b.p[2][0],
      a.p[2][0] * b.p[0][0] + a.p[2][1] * b.p[1][0] + a.p[2][2] * b.p[2][0],
      a.p[0][0] * b.p[0][1] + a.p[0][1] * b.p[1][1] + a.p[0][2] * b.p[2][1],
      a.p[1][0] * b.p[0][1] + a.p[1][1] * b.p[1][1] + a.p[1][2] * b.p[2][1],
      a.p[2][0] * b.p[0][1] + a.p[2][1] * b.p[1][1] + a.p[2][2] * b.p[2][1],
      a.p[0][0] * b.p[0][2] + a.p[0][1] * b.p[1][2] + a.p[0][2] * b.p[2][2],
      a.p[1][0] * b.p[0][2] + a.p[1][1] * b.p[1][2] + a.p[1][2] * b.p[2][2],
      a.p[2][0] * b.p[0][2] + a.p[2][1] * b.p[1][2] + a.p[2][2] * b.p[2][2]
      );
  }

  /**
   * @brief SSC (skew-symmetric) 3x3 matrix to matrix multiplication
   *
   * Contains less operations than common matrices multiplication
   * Can be used in vector-towards vector rotation
   * @see rotate_towards(vec3 subject, vec3 to)
   * @param[in] a 1st 3x3 SSC matrix
   * @param[in] b 2nd 3x3 SSC matrix
   */
  inline mat33 ssc_mul(mat33 a, mat33 b)
  {
    return make_mat33(
      a.p[0][1] * b.p[1][0] + a.p[0][2] * b.p[2][0],
      a.p[1][2] * b.p[2][0],
      a.p[2][1] * b.p[1][0],
      a.p[0][1] * b.p[1][1] + a.p[0][2] * b.p[2][1],
      a.p[1][0] * b.p[0][1] + a.p[1][2] * b.p[2][1],
      a.p[2][0] * b.p[0][1],
      a.p[0][1] * b.p[1][2],
      a.p[1][0] * b.p[0][2],
      a.p[2][0] * b.p[0][2] + a.p[2][1] * b.p[1][2]
      );
  }

  /// @brief 3x3 matrices sum
  inline mat33 operator + (mat33 a, mat33 b)
  {
    return make_mat33(
      a.p[0][0] + b.p[0][0], a.p[1][0] + b.p[1][0], a.p[2][0] + b.p[2][0],
      a.p[0][1] + b.p[0][1], a.p[1][1] + b.p[1][1], a.p[2][1] + b.p[2][1],
      a.p[0][2] + b.p[0][2], a.p[1][2] + b.p[1][2], a.p[2][2] + b.p[2][2]
      );
  }

  /// @brief Matrix to scalar multiplication
  inline mat33 operator * (mat33 a, point_t b)
  {
    return make_mat33(
      a.p[0][0] * b, a.p[1][0] * b, a.p[2][0] * b,
      a.p[0][1] * b, a.p[1][1] * b, a.p[2][1] * b,
      a.p[0][2] * b, a.p[1][2] * b, a.p[2][2] * b
      );
  }

  /// @brief Identity diagonal 3x3 matrix
  static const mat33 IDENTITY_33 = make_mat33(1, 0, 0, 0, 1, 0, 0, 0, 1);

  /**
   * @brief Produces 3x3 matrix of rotation source 3d vector to target 3d vector
   *
   * Function checks for parallel vectors but both of them should be normalized
   * @param[in] subject source 3d vector
   * @param[in] to target 3d vector
   * @return 3x3 rotation matrix
   * @warning Parameters should be normalized
   */
  mat33 rotate_towards(vec3 subject, vec3 to);

  /**
   * @brief Produces 3x3 matrix of rotation around axes
   * @param[in] x X-axis rotation angle
   * @param[in] y Y-axis rotation angle
   * @param[in] z Z-axis rotation angle
   * @return 3x3 rotation matrix
   */
  mat33 axis_rotation(float x, float y, float z);
}