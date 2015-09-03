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
 * This module contains basic types to represent a scene for form factors calculation.
 * Module also contains base type (system_t) for form factor calculation with table of virtual methods.
 */

#include "../cpu_emission/cpu_system.h"
#include "system.h"
#include <algorithm>
#include <stdlib.h>

namespace emission
{
  system_t* system_create(int type, ray_caster::system_t* ray_caster)
  {
    system_t* system = 0;
    switch (type)
    {
    case EMISSION_CPU:
      system = cpu_emission::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, ray_caster);

    return system;
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  int emitted_front(const task_t* task, int face_idx)
  {
    const float weight = task->weights[2 * face_idx];
    return weight > 0 ? std::max<int>(1, (int)(task->n_rays * (weight / task->total_weight))) : 0;
  }

  int emitted_rear(const task_t* task, int face_idx)
  {
    const float weight = task->weights[2 * face_idx + 1];
    return weight > 0 ? std::max<int>(1, (int)(task->n_rays * (weight / task->total_weight))) : 0;
  }

  task_t* task_create(int n_rays)
  {
    task_t* task = (task_t*)malloc(sizeof(task_t));
    task->n_rays = n_rays;
    task->total_weight = 0;
    task->weights = 0;
    task->rays = 0;
    return task;
  }

  void task_free(task_t* task)
  {
    if (task)
    {
      free(task->weights);
      ray_caster::task_free(task->rays);
    }
    free(task);
  }

  int system_init(system_t* system, ray_caster::system_t* ray_caster)
  {
    return system->methods->init(system, ray_caster);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, ray_caster::scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
