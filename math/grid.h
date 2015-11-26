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
* This module file contains grid definition and traversal algorithms.
*/

#pragma once

#include "types.h"

namespace math
{
  struct grid_coord_t
  {
    int x;
    int y;
  };

  inline bool operator==(const grid_coord_t& l, const grid_coord_t& r)
  {
    return l.x == r.x && l.y == r.y;
  }

  inline bool operator<(const grid_coord_t& l, const grid_coord_t& r)
  {
    if (l.x < r.x)
      return true;
    else if (l.x > r.x)
      return false;
    else
      return l.y < r.y;
  }

  struct grid_2d_t
  {
    vec3 base;
    vec3 side;
    int n_x;
    int n_y;
  };

  typedef bool(*grid_traversal_callback)(grid_coord_t p, void* param); // return true to stop

  /// @brief Travers ray (infinite ray) through grid.
  void grid_traverse(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param);

  /// @brief Put segment (finite ray) on grid.
  void grid_put(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param);

  void grid_rasterize(const grid_2d_t* grid, const triangle_t& t, grid_traversal_callback callback, void* param);
}
