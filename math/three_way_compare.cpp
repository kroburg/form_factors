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
* This module contains logic for three-way comparison which can be used to merge or process three data streams.
*/

#include "three_way_compare.h"

namespace math
{
  three_way_less_t three_way_less(int p0, int p1, int p2)
  {
    if (p0 < p1)
    {
      if (p0 < p2)
        return l3w_first_less;
      else if (p0 > p2)
        return l3w_third_less;
      else
        return l3w_second_greater;
    }
    else if (p1 < p0) // p0 > p1
    {
      if (p2 > p1)
        return l3w_second_less;
      else if (p2 < p1) // p1 > p2
        return l3w_third_less;
      else
        return l3w_first_greater;
    }
    else // p0 == p1
    {
      if (p2 < p0)
        return l3w_third_less;
      else if (p2 > p0)
        return l3w_third_greater;
      else
        return l3w_equal;
    }
  }

  two_way_less_t two_way_less(int p0, int p1)
  {
    if (p0 < p1)
      return l2w_first_less;
    else if (p0 > p1)
      return l2w_second_less;
    else
      return l2w_equal;
  }
}