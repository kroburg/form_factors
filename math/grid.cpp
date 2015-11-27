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
#include "triangle.h"
#include <stdlib.h>
#include <algorithm>

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

  struct boundary_t
  {
    int pivot;
    int min;
    int max;
  };

  struct boundary_less
  {
    bool operator()(const boundary_t& l, int r) const
    {
      return l.pivot < r;
    }

    bool operator()(int l, const boundary_t& r) const
    {
      return l < r.pivot;
    }

    bool operator()(const boundary_t& l, const boundary_t& r) const
    {
      return l.pivot < r.pivot;
    }
  };

  struct rasterizer_countour_t
  { 
    int size;
    boundary_t values[31];
  };

  bool collect_countour(grid_coord_t p, rasterizer_countour_t* c)
  {
    std::pair<boundary_t*, boundary_t*> range = std::equal_range(c->values, c->values + c->size, p.y, boundary_less());
    boundary_t& b = *range.first;
    if (range.first != range.second)
    { 
      b.min = std::min(b.min, p.x);
      b.max = std::max(b.max, p.x);
    }
    else
    {
      if (c->size != sizeof(c->values) / sizeof(boundary_t))
      {
        memmove(range.first + 1, range.first, sizeof(boundary_t) * (c->size - (range.first - c->values)));
        ++c->size;
        b = { p.y, p.x, p.x };
      }
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
    for (char i = 0; i != c.size; ++i)
    {
      const boundary_t& b = c.values[i];
      for (int x = b.min; x <= b.max; ++x)
        if (callback({ x, b.pivot }, param))
          return;
    }
  }

  void grid_rasterize(const grid_2d_t* grid, const triangle_t& t, grid_traversal_callback callback, void* param)
  {
    // @todo Replace std::map<> with more efficient collection.
    rasterizer_countour_t countour = { 0 };
    collector_countour(grid, countour, t);
    report_countour(countour, callback, param);
  }

  grid_2d_index_t* grid_make_index(const grid_2d_t* grid)
  {
    grid_2d_index_t* index = (grid_2d_index_t*)malloc(sizeof(grid_2d_index_t));
    index->n_x = grid->n_x;
    index->n_y = grid->n_y;
    index->table = (grid_triangles_list_t*)malloc(grid->n_x * grid->n_y * sizeof(grid_triangles_list_t));
    memset(index->table, 0, grid->n_x * grid->n_y * sizeof(grid_triangles_list_t));
    return index;
  }

  void grid_free_index(grid_2d_index_t* index)
  {
    if (index)
      free(index->table);
    free(index);
  }

  struct index_param_t
  {
    grid_2d_index_t* index;
    const triangle_t* triangle;
  };

  bool index_callback(grid_coord_t p, index_param_t* param)
  {
    grid_triangles_list_t& list = param->index->table[p.x + param->index->n_x * p.y];
    if (list.alloc - list.size == 0)
    {
      list.alloc += 8;
      list.triangles = (triangle_t const**)realloc(list.triangles, sizeof(triangle_t*) * list.alloc);
    }
    list.triangles[list.size++] = param->triangle;
    return false;
  }

  void grid_index_triangles(const grid_2d_t* grid, grid_2d_index_t* index, const triangle_t* triangles, int n_triangles)
  {
    for (int t = 0; t != n_triangles; ++t)
    {
      index_param_t param = { index, &triangles[t] };
      grid_rasterize(grid, triangles[t], (grid_traversal_callback)index_callback, &param);
    }
  }

  int grid_get_index_usage(const grid_2d_index_t* index)
  {
    int s = 0;
    for (int i = 0; i != index->n_x * index->n_y; ++i)
      if (index->table[i].size)
        ++s;
    return s;
  }

  void grid_draw_hist(int n_depth, const triangle_t* triangles, int n_triangles)
  {
    aabb_t aabb = triangles_aabb(triangles, n_triangles);
    aabb.max *= 1.0001f;
    printf("         | usage | per_tr\n");
    for (int i = 1; i <= n_depth; i *= 2)
    {
      grid_2d_t grid = { aabb.min, (aabb.max - aabb.min) / i, i, i };
      grid_2d_index_t* index = grid_make_index(&grid);
      grid_index_triangles(&grid, index, triangles, n_triangles);
      int usage = grid_get_index_usage(index);
      printf("%8d | %5.0f | %6.1f\n", i, 100 * (float)usage / i / i, (float)usage / n_triangles);
      grid_free_index(index);
    }
  }
}
