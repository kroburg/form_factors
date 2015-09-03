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
* This module contains thermal equation for Stefan-Boltzman radiation.
* Equation consider pre-calculated form factors matrix for meshes.
*/

#include "cpu_system.h"
#include "../math/triangle.h"
#include <float.h>
#include <cmath>
#include <limits>
#include <stdlib.h>
#include <cstring>

namespace sb_ff_te
{
  const float sigma = 5.670400e-8f;

  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_equation::system_t
  {    
    params_t params;

    subject::scene_t* scene;
    float* areas;

    // @todo Move form factors calculation of process?
    form_factors::system_t* ff_calculator;
    form_factors::scene_t ff_scene;
    form_factors::task_t* ff_task;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, params_t* params)
  {    
    system->params = *params;
    system->areas = 0;
    system->scene = 0;
    system->ff_calculator = params->form_factors_calculator;
    system->ff_task = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    system->scene = 0;

    free(system->areas);
    system->areas = 0;

    system->ff_calculator = 0;
    // @todo Looks like it is better to move task ownership to system to avoid such code.
    form_factors::task_free(system->ff_task);
    system->ff_task = 0;

    return THERMAL_EQUATION_OK;
  }

  /// @detail Precalculate form factors for given scene.
  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    system->scene = scene;
    
    build_meshes_areas(system->scene, &system->areas);

    system->ff_scene.n_faces = scene->n_faces;
    system->ff_scene.faces = scene->faces;
    system->ff_scene.n_meshes = scene->n_meshes;
    // @todo Use mesh size scene parameter.
    system->ff_scene.meshes = (form_factors::mesh_t*)scene->meshes;
    
    int r = 0;
    if ((r = form_factors::system_set_scene(system->ff_calculator, &system->ff_scene)) < 0)
      return r;

    system->ff_task = form_factors::task_create(&system->ff_scene, system->params.n_rays);

    if ((r = form_factors::system_calculate(system->ff_calculator, system->ff_task)) < 0)
      return r;

    return THERMAL_EQUATION_OK;
  }

  /**
  *  @brief Calculates thermal flow for a given scene.
  *
  *  Uses once precalculated (during set_scene() call) form factors.
  */
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    //@todo Note n^2 complexity.
    const int n_meshes = system->scene->n_meshes;
    for (int m = 0; m != n_meshes; ++m)
    {
      const int material_idx = system->scene->meshes[m].material_idx;
      const subject::material_t& material = mesh_material(system->scene, m);
      const float T = task->temperatures[m];
      float power = sigma * (T * T * T * T) * system->areas[m];
      float emission = material.front.emissivity * power + material.rear.emissivity * power;
      task->emission[m] += emission;
      for (int n = 0; n != n_meshes; ++n)
      {
        task->absorption[n] += emission * system->ff_task->form_factors[m * n_meshes + n];
      }
    }

    return THERMAL_EQUATION_OK;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_equation::system_methods_t methods =
  {
    (int(*)(thermal_equation::system_t* system, void* params))&init,
    (int(*)(thermal_equation::system_t* system))&shutdown,
    (int(*)(thermal_equation::system_t* system, subject::scene_t* scene))&set_scene,
    (int(*)(thermal_equation::system_t* system, thermal_equation::task_t* task))&calculate,
  };

  thermal_equation::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

}
