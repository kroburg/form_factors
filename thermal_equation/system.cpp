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
* This module contains basic types to represent a thermal equation.
*/

#include "system.h"
#include <stdlib.h>
#include <cstring>

namespace thermal_equation
{
  void system_free(system_t* system)
  {
    system_shutdown(system);
    free(system);
  }

  task_t* task_create(subject::scene_t* scene)
  {
    task_t* task = (task_t*)malloc(sizeof(task_t));
    task->temperatures = 0;
    const int mem_size = sizeof(float) * scene->n_meshes;
    task->emission = (float*)malloc(mem_size);
    task->absorption = (float*)malloc(mem_size);
    memset(task->emission, 0, mem_size);
    memset(task->absorption, 0, mem_size);
    return task;
  }

  void task_free(task_t* task)
  {
    if (task)
    {
      free(task->emission);
      free(task->absorption);
    }
    free(task);
  }

  int system_init(system_t* system, void* params)
  {
    return system->methods->init(system, params);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, subject::scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_calculate(system_t* system, task_t* task)
  {
    return system->methods->calculate(system, task);
  }
}
