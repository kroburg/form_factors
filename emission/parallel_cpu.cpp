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
* This module contains CPU single-threaded implementation of emission calculator.
* Calculator is capable of work with CPU or GPU ray caster implementation.
*/

#include "parallel_cpu.h"
#include "../ray_caster/system.h"
#include "../math/operations.h"
#include "../math/triangle.h"
#include "../math/mat.h"
#include <float.h>
#include <random>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace parallel_emission_cpu
{
  struct cpu_system_t : emission::system_t
  {
    params_t params;
    math::vec3 basis_x;
    math::vec3 basis_y;
    
    ray_caster::scene_t* scene;
    ray_caster::system_t* ray_caster;    

    // Mersenne's twister uniformly distributed [0, 1) generators.
    std::mt19937 TPGenX;
    std::mt19937 TPGenY;

    // [0, 1)
    std::uniform_real_distribution<float> Distr_X;
    std::uniform_real_distribution<float> Distr_Y;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, ray_caster::system_t* ray_caster, params_t* params)
  {
    system->params = *params;
    math::mat33 rotation = math::rotate_towards(math::make_vec3(0, 0, 1), params->direction);
    system->basis_x = rotation * math::make_vec3(params->width, 0, 0);
    system->basis_y = rotation * math::make_vec3(0, params->height, 0);

    system->scene = 0;
    system->ray_caster = ray_caster;

    system->TPGenX = std::mt19937(1);
    system->TPGenY = std::mt19937(2);
    system->Distr_X = std::uniform_real_distribution<float>(0, 1);
    system->Distr_Y = std::uniform_real_distribution<float>(0, 1);
    return EMISSION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;
    return EMISSION_OK;
  }

  /// @brief Sets loaded scene (polygons in meshes) for calculator and associated ray caster.
  int set_scene(cpu_system_t* system, ray_caster::scene_t* scene)
  {
    if (scene == 0 || scene->n_faces == 0)
      return -EMISSION_ERROR;

    system->scene = scene;

    int r = 0;
    if ((r = ray_caster::system_set_scene(system->ray_caster, system->scene)) < 0)
      return r;

    if ((r = ray_caster::system_prepare(system->ray_caster)) < 0)
      return r;

    return EMISSION_OK;
  }


  int calculate_n_rays(emission::task_t* task, int n_faces)
  {
    int result = 0;
    for (int f = 0; f != n_faces; ++f)
    {
      const int face_rays_front = emitted_front(task, f);
      const int face_rays_rear = emitted_rear(task, f);
      result += face_rays_front + face_rays_rear;
    }
    return result;
  }

  /// @brief Generates uniformly distributed point on plane.
  math::vec3 pick_point(cpu_system_t* system)
  { 
    return system->basis_x * system->Distr_X(system->TPGenX)
      + system->basis_y * system->Distr_Y(system->TPGenY)
      + system->params.origin;
  }

  /// @brief Creates task with n_rays random generated rays.
  ray_caster::task_t* make_caster_task(cpu_system_t* system, emission::task_t* task)
  {
    ray_caster::task_t* ray_caster_task = ray_caster::task_create(task->n_rays);
    for (int r = 0; r != task->n_rays; ++r)
    {
      math::vec3 origin = pick_point(system);
      math::vec3 direction = system->params.direction;
      ray_caster_task->ray[r] = { origin, direction };
    }

    return ray_caster_task;
  }

  int calculate(cpu_system_t* system, emission::task_t* task)
  {
    if (task->rays)
    {
      // @todo Provide realloc
      ray_caster::task_free(task->rays);
      task->rays = 0;
    }
    task->rays = make_caster_task(system, task);
    int r = 0;
    if ((r = ray_caster::system_cast(system->ray_caster, task->rays)) < 0)
      return r;

    return EMISSION_OK;
  }

  const emission::system_methods_t methods =
  {
    (int(*)(emission::system_t* system, ray_caster::system_t* ray_caster, void* params))&init,
    (int(*)(emission::system_t* system))&shutdown,
    (int(*)(emission::system_t* system, ray_caster::scene_t* scene))&set_scene,
    (int(*)(emission::system_t* system, emission::task_t* task))&calculate,
  };

  emission::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }
}
