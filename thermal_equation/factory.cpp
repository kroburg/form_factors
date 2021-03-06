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
#include "conductive_cpu.h"
#include "form_factors.h"
#include "radiance_cpu.h"
#include "heat_source_cpu.h"
#include "parallel_rays_cpu.h"

namespace thermal_equation
{

  system_t* system_create(int type, void* params)
  {
    system_t* system = 0;
    switch (type)
    {
    case THERMAL_EQUATION_SB_FF_CPU:
      system = form_factors_equation::system_create();
      break;

    case THERMAL_EQUATION_RADIANCE_CPU:
      system = radiance_equation::system_create();
      break;

    case THERMAL_EQUATION_CONDUCTIVE_CPU:
      system = conductive_equation::system_create();
      break;

    case THERMAL_EQUATION_HEAT_SOURCE_CPU:
      system = heat_source_equation::system_create();
      break;

    case THERMAL_EQUATION_DISNTANT_SOURCE_CPU:
      system = parallel_rays_cpu::system_create();
      break;

    default:
      return 0;
    }

    system_init(system, params);

    return system;
  }
}