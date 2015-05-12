#include "mat.h"

namespace math
{
  mat33 rotate_towards(vec3 subject, vec3 to)
  {
    subject = normalize(subject);
    vec3 v = cross(subject, to);
    point_t s2 = dot(v, v);
    point_t c = dot(subject, to); // TODO: Normalize?

    mat33 rot = IDENTITY_33;

    if (s2 > FLT_MIN)
    {
      mat33 ssc = make_mat33(0, v.z, -v.y, -v.z, 0, v.x, v.y, -v.x, 0);
      mat33 ssc2 = ssc_mul(ssc, ssc);
      ssc2 = ssc2 * ((1 - c) / s2);
      rot = rot + ssc + ssc2;
    }

    return rot;
  }

  math::mat33 axis_rotation(float x, float y, float z)
  {
    math::mat33 rx = math::make_mat33(1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
    math::mat33 ry = math::make_mat33(cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
    math::mat33 rz = math::make_mat33(cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);
    return rx * ry * rz;
  }
}