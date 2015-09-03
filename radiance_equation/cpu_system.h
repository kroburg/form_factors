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
 * This module contains thermal equation for Stefan-Boltzman radiation using Monte-Carlo method.
 * Calculate faces weights based on power emission.
 * Calculate power emission/absorption on each step.
 */

#pragma once

#include "../thermal_equation/system.h"
#include "../emission/system.h"

namespace radiance_equation
{
  struct params_t
  {
    emission::system_t* emitter;
    int n_rays;
  };

  thermal_equation::system_t* system_create();
}
