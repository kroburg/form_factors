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

#pragma once

namespace math
{
  /// @detail Value bits represent index of iterator which require increment.
  enum three_way_less_t
  { 
    l3w_first_less = 1 << 0, // And nothing known about relationship between 2 and 3.
    l3w_second_less = 1 << 1,
    l3w_third_less = 1 << 2,
    l3w_first_greater = l3w_second_less | l3w_third_less, // (2 == 3) < 1
    l3w_second_greater = l3w_first_less | l3w_third_less, // (1 == 3) < 2
    l3w_third_greater = l3w_first_less | l3w_second_less, // (1 == 2) < 3      
    l3w_equal = l3w_first_less | l3w_second_less | l3w_third_less
  };

  enum two_way_less_t
  {
    l2w_first_less = 1 << 0,
    l2w_second_less = 1 << 1,
    l2w_equal = l2w_first_less | l2w_second_less
  };

  three_way_less_t three_way_less(int p0, int p1, int p2);
  two_way_less_t two_way_less(int p0, int p1);
}