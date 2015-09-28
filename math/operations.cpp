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

#include "operations.h"

namespace math
{
  bool operator < (const vec3& l, const vec3& r)
  {
    const math::vec3 size = max(abs(l), abs(r));
    math::vec3 diff = (l - r) / size;
    if (diff.x < -FLT_EPSILON)
      return true;
    else if (diff.x > FLT_EPSILON)
      return false;
    else if (diff.y < -FLT_EPSILON)
      return true;
    else if (diff.y > FLT_EPSILON)
      return false;
    else if (diff.z < -FLT_EPSILON)
      return true;
    else
      return false;
  }
}