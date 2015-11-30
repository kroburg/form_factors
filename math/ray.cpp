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

#include "ray.h"
#include "operations.h"

namespace math
{
  /// See for details http ://geomalgorithms.com/a05-_intersect-1.html
  bool ray_intersect_segment(ray_t p, ray_t q, float& s)
  {
    vec3 v = q.direction - q.origin;
    vec3 w = p.origin - q.origin;
    vec3 u = p.direction - p.origin;

    float d = (v.x * u.y - v.y * u.x);

    s = (v.y * w.x - v.x * w.y) / d;
    float t = (u.x * w.y - u.y * w.x) / -d;

    return t >= 0 && t <= 1 && s >= 0;
  }
}
