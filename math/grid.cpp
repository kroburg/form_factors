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

#include "grid.h"
#include "operations.h"
#include "assert.h"

#include <map>

namespace math
{
  void grid_traverse(const grid_2d_t* grid, ray_t ray, bool is_segment, grid_traversal_callback callback, void* param)
  { 
    vec3 u = ray.origin - grid->base;
    vec3 v = ray.direction - ray.origin;
    // @todo Understand abs() logic - is it really required, or t can be negative.
    vec3 t_delta = abs(grid->side / v); // 1 / velocity -> cells per v step
    vec3 base_distance = make_vec3(v.x < 0 ? 0 : 1.f, v.y < 0 ? 0 : 1.f, 0);
    vec3 t = abs(t_delta * (base_distance - fraction(u / grid->side))); // |(base_distance - fraction) is relative distance to next row.

    int step_x = v.x < 0 ? -1 : 1;
    int step_y = v.y < 0 ? -1 : 1;

    grid_coord_t stop = { v.x < 0 ? -1 : grid->n_x, v.y < 0 ? -1 : grid->n_y };
    grid_coord_t p = { (int)(u.x / grid->side.x), (int)(u.y / grid->side.y) };

    assert(p.x >= 0 && p.x < grid->n_x);
    assert(p.y >= 0 && p.y < grid->n_y);

    while (true)
    {
      if (callback(p, param))
        return;

      if (is_segment && (t.x > 1 && t.y > 1))
        return;

      const bool do_x = t.x <= t.y;
      const bool do_y = t.y <= t.x;

      if (do_x)
      {
        p.x += step_x;
        if (p.x == stop.x)
          return;
        t.x += t_delta.x;
      }

      if (do_y)
      {
        p.y += step_y;
        if (p.y == stop.y)
          return;
        t.y += t_delta.y;
      }
    }
  }

  void grid_traverse(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param)
  {
    grid_traverse(grid, ray, false, callback, param);
  }

  void grid_put(const grid_2d_t* grid, ray_t ray, grid_traversal_callback callback, void* param)
  {
    grid_traverse(grid, ray, true, callback, param);
  }

  struct x_boundary_t
  {
    int min;
    int max;
  };

  typedef std::map<int, x_boundary_t> rasterizer_countour_t;

  bool collect_countour(int x, int y, rasterizer_countour_t* countour)
  {
    rasterizer_countour_t::iterator found = countour->find(y);
    x_boundary_t& b = (*countour)[y];
    if (found == countour->end())
    { 
      b.min = x;
      b.max = x;
    }
    else
    {
      b.min = std::min(b.min, x);
      b.max = std::max(b.max, x);
    }

    return false;
  }

  void collector_countour(const grid_2d_t* grid, rasterizer_countour_t& c, const triangle_t& t)
  {
    grid_put(grid, { t.points[0], t.points[1] }, (grid_traversal_callback)&collect_countour, &c);
    grid_put(grid, { t.points[0], t.points[2] }, (grid_traversal_callback)&collect_countour, &c);
    grid_put(grid, { t.points[1], t.points[2] }, (grid_traversal_callback)&collect_countour, &c);
  }

  void report_countour(const rasterizer_countour_t& c, grid_traversal_callback callback, void* param)
  {
    for (rasterizer_countour_t::const_iterator iter = c.begin(); iter != c.end(); ++iter)
    {
      const x_boundary_t& b = iter->second;
      for (int x = b.min; x <= b.max; ++x)
        if (callback({ x, iter->first }, param))
          return;
    }
  }

  void grid_rasterize(const grid_2d_t* grid, const triangle_t& t, grid_traversal_callback callback, void* param)
  {
    // @todo Replace std::map<> with more efficient collection.
    rasterizer_countour_t countour;
    collector_countour(grid, countour, t);
    report_countour(countour, callback, param);
  }
}
