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
    math::mat33 rx = math::make_mat33(1, 0, 0, 0, cosf(x), -sinf(x), 0, sinf(x), cosf(x));
    math::mat33 ry = math::make_mat33(cosf(y), 0, sinf(y), 0, 1, 0, -sinf(y), 0, cosf(y));
    math::mat33 rz = math::make_mat33(cosf(z), -sinf(z), 0, sinf(z), cosf(z), 0, 0, 0, 1);
    return rx * ry * rz;
  }
}