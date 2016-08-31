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
#include"ray.h"
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <string.h>

namespace math
{
  bool rebase_if_out_of_bound(const grid_2d_t* grid, ray_t& ray)
  {
    vec3 u = ray.origin - grid->base;
    vec3 h = u - grid->size;
    vec3 v = ray.direction - ray.origin;

    bool out_of_bound = u.x < 0 || h.x > 0 || u.y < 0 || h.y > 0;
    if (!out_of_bound)
      return true;

    ray_t plane_ray = make_ray(u, ray.direction - grid->base);

    float s1;
    float s2;
    float s3;
    float s4;
    bool i1 = ray_intersect_segment(plane_ray, make_ray(make_vec3(0, 0, 0), make_vec3(grid->size.x, 0, 0)), s1);
    bool i2 = ray_intersect_segment(plane_ray, make_ray(make_vec3(0, 0, 0), make_vec3(0, grid->size.y, 0)), s2);
    bool i3 = ray_intersect_segment(plane_ray, make_ray(make_vec3(0, grid->size.y, 0), make_vec3(grid->size.x, grid->size.y, 0)), s3);
    bool i4 = ray_intersect_segment(plane_ray, make_ray(make_vec3(grid->size.x, 0, 0), make_vec3(grid->size.x, grid->size.y, 0)), s4);

    if (!i1 && !i2 && !i3 && !i4)
      return false;

    float s = FLT_MAX;
    if (i1) s = std::min(s, s1);
    if (i2) s = std::min(s, s2);
    if (i3) s = std::min(s, s3);
    if (i4) s = std::min(s, s4);

    vec3 shift = v * (s + 0.00001f);
    ray.origin += shift;
    ray.direction += shift;
    
    return true;
  }

  void grid_traverse(const grid_2d_t* grid, ray_t ray, bool is_segment, grid_traversal_callback callback, void* param)
  {
    if (!rebase_if_out_of_bound(grid, ray))
      return;

    vec3 u = ray.origin - grid->base;
    vec3 v = ray.direction - ray.origin;
    grid_coord_t p = { (int)(u.x / grid->side.x), (int)(u.y / grid->side.y) };
    
    // @todo Understand abs() logic - is it really required, or t can be negative.
    vec3 t_delta = abs(grid->side / v); // 1 / velocity -> cells per v step
    vec3 base_distance = make_vec3(v.x < 0 ? 0 : 1.f, v.y < 0 ? 0 : 1.f, 0);
    vec3 t = abs(t_delta * (base_distance - fraction(u / grid->side))); // |(base_distance - fraction) is relative distance to next row.

    int step_x = v.x < 0 ? -1 : 1;
    int step_y = v.y < 0 ? -1 : 1;

    grid_coord_t stop = { v.x < 0 ? -1 : grid->n_x, v.y < 0 ? -1 : grid->n_y };

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
    boundary_t values[grid_rasterizer_countour_buffer_size];
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
    // @todo Collect countour by axis with minimal object dimension.
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
    index->cells = (grid_cell_t*)malloc(grid->n_x * grid->n_y * sizeof(grid_cell_t));
    memset(index->cells, 0, grid->n_x * grid->n_y * sizeof(grid_cell_t));
    index->triangles = 0;
    index->n_triangles = 0;
    return index;
  }

  void grid_free_index(grid_2d_index_t* index)
  {
    if (index)
    {
      if (index->cells)
        free(index->cells);
      if (index->triangles)
        free(index->triangles);
    }
    free(index);
  }

  struct grid_2d_index_builder_t
  {
    struct node
    {
      int size;
      int alloc;
      int* triangles;
    };

    int n_x;
    int n_y;
    node* table;
  };

  void grid_index_linearize(grid_2d_index_builder_t* builder, grid_2d_index_t* index)
  {
    int count = 0;
    for (int i = 0; i != index->n_x * index->n_y; ++i)
    {
      count += builder->table[i].size;
    }

    int* linear = (int*)malloc(count * sizeof(int));
    index->triangles = linear;
    index->n_triangles = count;

    for (int i = 0, l = 0; i != index->n_x * index->n_y; ++i)
    {
      grid_2d_index_builder_t::node& source = builder->table[i];
      if (!source.size)
        continue;
      
      memcpy(&linear[l], source.triangles, source.size * sizeof(int));
      free(source.triangles);

      grid_cell_t& target = index->cells[i];
      target.offset = l;
      target.count = source.size;
      l += source.size;
    }
  }

  struct index_param_t
  {
    grid_2d_index_builder_t* index;
    int triangle;
  };

  bool index_callback(grid_coord_t p, index_param_t* param)
  {
    grid_2d_index_builder_t::node& list = param->index->table[p.x + param->index->n_x * p.y];
    if (list.alloc - list.size == 0)
    {
      list.alloc += 8;
      list.triangles = (int*)realloc(list.triangles, sizeof(int) * list.alloc);
    }
    list.triangles[list.size++] = param->triangle;
    return false;
  }

  void grid_index_triangles(const grid_2d_t* grid, grid_2d_index_t* index, const triangle_t* triangles, int n_triangles)
  {
    grid_2d_index_builder_t builder;
    builder.n_x = index->n_x;
    builder.n_y = index->n_y;
    builder.table = (grid_2d_index_builder_t::node*)calloc(builder.n_x * builder.n_y, sizeof(grid_2d_index_builder_t::node));
    
    for (int t = 0; t != n_triangles; ++t)
    {
      index_param_t param = { &builder, t };
      grid_rasterize(grid, triangles[t], (grid_traversal_callback)index_callback, &param);
    }

    grid_index_linearize(&builder, index);
    free(builder.table);
  }

  int grid_get_index_usage(const grid_2d_index_t* index)
  {
    int s = 0;
    for (int i = 0; i != index->n_x * index->n_y; ++i)
      if (index->cells[i].count)
        ++s;
    return s;
  }

  void grid_draw_hist(int n_depth, const triangle_t* triangles, int n_triangles)
  {
    triangles_analysis_t stat = triangles_analyze(triangles, n_triangles);
    grid_2d_t optimal = grid_deduce_optimal(stat);
    printf("         | usage | per_tr\n");
    for (int i = 1; i <= n_depth; i *= 2)
    {
      grid_2d_t grid = make_grid(stat.aabb.min, stat.aabb.max - stat.aabb.min, i, i);
      grid_2d_index_t* index = grid_make_index(&grid);
      grid_index_triangles(&grid, index, triangles, n_triangles);
      int usage = grid_get_index_usage(index);
      printf("%8d | %5.0f | %6.1f\n", i, 100 * (float)usage / i / i, (float)usage / n_triangles);
      grid_free_index(index);
    }
  }

  grid_2d_t grid_deduce_optimal(triangles_analysis_t stat)
  {
    vec3 volume = stat.aabb.max - stat.aabb.min;
    float total_area = volume.x * volume.y;
    int n_avg_side = sqrtf(total_area / (stat.average_area * 1.5f));
    int n_avg_max_side = sqrtf(grid_rasterizer_countour_buffer_size * total_area / (2.5f * stat.average_area_xy));
    int n_max_side = sqrtf(grid_rasterizer_countour_buffer_size * total_area / (2.5f * stat.max_area_xy));

    int side = std::min(n_max_side, n_avg_max_side);
    return make_grid(stat.aabb.min, volume, side, side);
  }
}
