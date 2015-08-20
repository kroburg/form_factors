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
* This module contains thermal solution using Adams integration method implemented on CPU.
*/

#include "cpu_adams.h"
#include "../thermal_equation/system.h"
#include <stdlib.h>

namespace cpu_adams
{
  /// @brief Extended base system_t (C-style polymorphism)
  struct cpu_system_t : thermal_solution::system_t
  {
    thermal_solution::params_t params;
    thermal_solution::scene_t* scene;
  };

  /// @brief Initializes system with given ray caster after creation.
  int init(cpu_system_t* system, thermal_solution::params_t* params)
  {
    system->params = *params;
    system->scene = 0;
    
    return THERMAL_SOLUTION_OK;
  }

  /// @brief Shutdowns calculator system prior to free memory.
  int shutdown(cpu_system_t* system)
  {
    thermal_solution::scene_free(system->scene);
    system->scene = 0;

    return THERMAL_SOLUTION_OK;
  }

  int set_scene(cpu_system_t* system, thermal_solution::scene_t* scene)
  {
    system->scene = scene;

    int r = 0;
    for (int i = 0; i != system->params.n_equations; ++i)
    {
      thermal_equation::system_t* equation = &system->params.equations[i];
      if ((r = thermal_equation::system_set_scene(equation, scene)) < 0)
        return r;
    }

    return THERMAL_SOLUTION_OK;
  }

  /**
  *  @brief Calculates thermal solution step for a given scene.
  */
  int calculate(cpu_system_t* system, thermal_solution::task_t* task)
  {
    return THERMAL_SOLUTION_ERROR;
  }

  /// @brief Creates virtual methods table from local methods.
  const thermal_solution::system_methods_t methods =
  {
    (int(*)(thermal_solution::system_t* system, thermal_solution::params_t* params))&init,
    (int(*)(thermal_solution::system_t* system))&shutdown,
    (int(*)(thermal_solution::system_t* system, thermal_solution::scene_t* scene))&set_scene,
    (int(*)(thermal_solution::system_t* system, thermal_solution::task_t* task))&calculate,
  };

  thermal_solution::system_t* system_create()
  {
    cpu_system_t* s = (cpu_system_t*)malloc(sizeof(cpu_system_t));
    s->methods = &methods;
    return s;
  }

}
