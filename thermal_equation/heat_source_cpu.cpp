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
* This module contains thermal equation for heat sources and sinks.
*/

#include "heat_source_cpu.h"
#include <cstdlib>

namespace heat_source_equation
{
  struct cpu_system_t : thermal_equation::system_t
  {
    params_t params;
  };

  int init(cpu_system_t* system, params_t* params)
  {
    system->params = *params;

    return THERMAL_EQUATION_OK;
  }

  int shutdown(cpu_system_t* system)
  {
    return THERMAL_EQUATION_OK;
  }

  int set_scene(cpu_system_t* system, subject::scene_t* scene)
  {
    return THERMAL_EQUATION_OK;
  }
  
  int calculate(cpu_system_t* system, thermal_equation::task_t* task)
  {
    const params_t& params = system->params;
    for (int i = 0; i != params.n_sources; ++i)
    {
      const heat_source_t& source = params.sources[i];
      if (source.power > 0)
        task->emission[source.mesh_idx] += source.power;
      else
        task->emission[source.mesh_idx] += -source.power;
    }

    return THERMAL_EQUATION_OK;
  }

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
