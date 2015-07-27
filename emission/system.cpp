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

#include "system.h"
#include <stdlib.h>

namespace emission
{
  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)malloc(sizeof(scene_t));
    *s = { 0, 0, 0, 0 };
    return s;
  }

  void scene_free(scene_t* scene)
  {
    if (scene)
    {
      free(scene->faces);
      free(scene->meshes);
    }
    free(scene);
  }

  system_t* system_create(int type, ray_caster::system_t* ray_caster, calculate_weights weights)
  {
    system_t* system = 0;
    switch (type)
    {
    case EMISSION_CPU:
      //system = cpu_form_factors::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, ray_caster, weights);

    return system;
  }

  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  task_t* task_create(scene_t* scene, int n_rays)
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

  int system_init(system_t* system, ray_caster::system_t* ray_caster, calculate_weights weights)
  {
    return system->methods->init(system, ray_caster, weights);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
