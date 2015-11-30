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

#include "operations.h"
#include "types.h"
#include "triangle.h"

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
    vec3 size;
    int n_x;
    int n_y;
  };

  inline grid_2d_t make_grid(vec3 base, vec3 size, int n_x, int n_y)
  {
    return{ base, size / make_vec3((float)n_x, (float)n_y, 0), size, n_x, n_y };
  }

  struct grid_cell_t
  {
    int count;
    int* triangles;
  };

  struct grid_2d_index_t
  {
    int n_x;
    int n_y;
    grid_cell_t* cells;
    int* triangles;
    int n_triangles;
  };

  grid_2d_index_t* grid_make_index(const grid_2d_t* grid);
  void grid_free_index(grid_2d_index_t* index);

  typedef bool(*grid_traversal_callback)(grid_coord_t p, void* param); // return true to stop

  /// @brief Travers ray (infinite ray) through grid.
  void grid_traverse(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param);

  /// @brief Put segment (finite ray) on grid.
  void grid_put(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param);

  const int grid_rasterizer_countour_buffer_size = 41;
  void grid_rasterize(const grid_2d_t* grid, const triangle_t& t, grid_traversal_callback callback, void* param);

  void grid_index_triangles(const grid_2d_t* grid, grid_2d_index_t* index, const triangle_t* triangles, int n_triangles);
  void grid_draw_hist(int n_depth, const triangle_t* triangles, int n_triangles);

  grid_2d_t grid_deduce_optimal(triangles_analysis_t stat);
}
