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
 * This module contains basic types to represent a scene for ray caster calculation.
 * Module also contains base type (system_t) for ray caster with table of virtual methods.
 */

#include "system.h"
#include "../cpuRayCaster/cpu_system.h"
#include "../cudaRayCaster/cuda_system.h"
#include <stdlib.h>

namespace ray_caster
{
  scene_t* scene_create()
  {
    scene_t* s = (scene_t*)malloc(sizeof(scene_t));
    *s = { 0, 0 };
    return s;
  }

  void scene_free(scene_t* scene)
  {
    if (scene)
      free(scene->faces);
    free(scene);
  }

  task_t* task_create(int n_rays)
  {
    task_t* task = (task_t*) malloc(sizeof(task_t));
    task->n_tasks = n_rays;
    task->ray = (math::ray_t*)     malloc(n_rays * sizeof(math::ray_t));
    task->hit_face = (face_t**)    malloc(n_rays * sizeof(ray_caster::face_t*));
    task->hit_point = (math::vec3*)malloc(n_rays * sizeof(math::vec3));
    return task;
  }

  void task_free(task_t* task)
  {
    if (task)
    {
      free(task->ray);
      free(task->hit_face);
      free(task->hit_point);
      free(task);
    }
  }

  system_t* system_create(int type)
  {
    system_t* system = 0;
    switch (type)
    {
    case RAY_CASTER_SYSTEM_CPU:
      system = cpu_ray_caster::system_create();
      break;

    case RAY_CASTER_SYSTEM_CUDA:
      system = cuda_ray_caster::system_create();
      break;
    }    

    system_init(system);
    
    return system;
  }

  system_t* system_create_default()
  {
    return system_create(RAY_CASTER_SYSTEM_CPU);
  }

  void system_free(system_t* system)
  { 
    system_shutdown(system);
    free(system);
  }

  int system_init(system_t* system)
  {
    return system->methods->init(system);
  }

  int system_shutdown(system_t* system)
  {
    return system->methods->shutdown(system);
  }

  int system_set_scene(system_t* system, scene_t* scene)
  {
    return system->methods->set_scene(system, scene);
  }

  int system_prepare(system_t* system)
  {
    return system->methods->prepare(system);
  }

  int system_cast(system_t* system, task_t* task)
  {
    return system->methods->cast(system, task);
  }
}
