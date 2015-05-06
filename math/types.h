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
}
