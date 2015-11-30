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
*  Grid raycaster with geometry merged by Z axis.
*/

#include "zgrid_cpu.h"
#include "../math/grid.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include <limits>
#include <stdlib.h>

namespace ray_caster_zgrid_cpu
{
  struct cpu_system_t : ray_caster::system_t
  {
    ray_caster::scene_t* scene;
    math::grid_2d_t grid;
    math::grid_2d_index_t* index;
  };

  int init(cpu_system_t* system)
  {
    system->scene = 0;
    system->index = 0;

    return RAY_CASTER_OK;
  }

  int shutdown(cpu_system_t* system)
  { 
    math::grid_free_index(system->index);
    return RAY_CASTER_OK;
  }

  int set_scene(cpu_system_t* system, ray_caster::scene_t* scene)
  {
    system->scene = scene;
    return RAY_CASTER_OK;
  }

  int prepare(cpu_system_t* system)
  {
    if (system->scene == 0 || system->scene->n_faces == 0)
      return -RAY_CASTER_ERROR;
    
    math::triangles_analysis_t stat = triangles_analyze(system->scene->faces, system->scene->n_faces);
    system->grid = grid_deduce_optimal(stat);
    system->index = grid_make_index(&system->grid);
    grid_index_triangles(&system->grid, system->index, system->scene->faces, system->scene->n_faces);
    
    return RAY_CASTER_OK;
  }

  struct callback_param_t
  {
    cpu_system_t* system;
    math::ray_t ray;
    math::point_t min_distance;
    int hit_face;
    math::vec3 hit_point;
  };

  bool grid_callback(math::grid_coord_t p, callback_param_t* param)
  {
    cpu_system_t* system = param->system;
    math::grid_2d_index_t* index = param->system->index;

    const math::grid_cell_t& cell = index->cells[p.x + index->n_x * p.y];
    // @todo Sort geometry in z-order?
    for (int i = 0; i != cell.count; ++i)
    {
      const int f = index->triangles[cell.offset + i];
      math::triangle_t triangle = system->scene->faces[f];
      math::vec3 point;
      const math::ray_t& ray = param->ray;
      int check_result = triangle_intersect(ray, triangle, &point);
      if (check_result == TRIANGLE_INTERSECTION_UNIQUE)
      {
        // Check if we hit geometry in the current cell.
        math::vec3 locator = (point - system->grid.base) / system->grid.side;
        if (p.x != (int)locator.x || p.y != (int)locator.y)
          continue;

        math::vec3 space_distance = point - ray.origin;
        math::point_t new_distance = dot(space_distance, space_distance);
        if (new_distance < param->min_distance)
        {
          param->min_distance = new_distance;
          param->hit_face = f;
          param->hit_point = point;
        }
      }
    }

    // Don't continue traversal if we hit something.
    return param->hit_face != -1;
  }

  int cast(cpu_system_t* system, ray_caster::task_t* task)
  {
    for (int t = 0; t != task->n_tasks; ++t)
    {
      task->hit_face[t] = 0;

      math::ray_t ray = task->ray[t];
      callback_param_t param = { system, ray, std::numeric_limits<math::point_t>::max(), -1 };
      math::grid_traverse(&system->grid, ray, (math::grid_traversal_callback)grid_callback, &param);
      if (param.hit_face != -1)
      {
        task->hit_face[t] = &system->scene->faces[param.hit_face];
        task->hit_point[t] = param.hit_point;
      }
    }

    return RAY_CASTER_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const ray_caster::system_methods_t methods =
  {
    (int(*)(ray_caster::system_t* system))&init,
    (int(*)(ray_caster::system_t* system))&shutdown,
    (int(*)(ray_caster::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(ray_caster::system_t* system))&prepare,
    (int(*)(ray_caster::system_t* system, ray_caster::task_t* task))&cast,
  };

  /// @brief Creates base system for ray caster.
  ray_caster::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
