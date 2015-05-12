#pragma once

#include "types.h"
#include "operations.h"
#include <float.h>

namespace math
{
  inline vec3 operator * (mat33 mat, vec3 vec)
  {
    return make_vec3(
      (mat.p[0][0] * vec.x) + (mat.p[1][0] * vec.y) + (mat.p[2][0] * vec.z),
      (mat.p[0][1] * vec.x) + (mat.p[1][1] * vec.y) + (mat.p[2][1] * vec.z),
      (mat.p[0][2] * vec.x) + (mat.p[1][2] * vec.y) + (mat.p[2][2] * vec.z)
      );
  }

  inline mat33 operator * (mat33 a, mat33 b){
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

  inline mat33 operator + (mat33 a, mat33 b)
  {
    return make_mat33(
      a.p[0][0] + b.p[0][0], a.p[1][0] + b.p[1][0], a.p[2][0] + b.p[2][0],
      a.p[0][1] + b.p[0][1], a.p[1][1] + b.p[1][1], a.p[2][1] + b.p[2][1],
      a.p[0][2] + b.p[0][2], a.p[1][2] + b.p[1][2], a.p[2][2] + b.p[2][2]
      );
  }

  inline mat33 operator * (mat33 a, point_t b) {
    return make_mat33(
      a.p[0][0] * b, a.p[1][0] * b, a.p[2][0] * b,
      a.p[0][1] * b, a.p[1][1] * b, a.p[2][1] * b,
      a.p[0][2] * b, a.p[1][2] * b, a.p[2][2] * b
      );
  }

  static const mat33 IDENTITY_33 = make_mat33(1, 0, 0, 0, 1, 0, 0, 0, 1);

  mat33 rotate_towards(vec3 subject, vec3 to);

  mat33 axis_rotation(float x, float y, float z);
}